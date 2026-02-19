"""
Loads the `open_qa` subset of databricks-dolly-15k, rewrites each answer in
Gen Z slang via a local vLLM instance, and saves the result as
`genz_dataset.jsonl`.

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import opik
from datasets import load_dataset
from openai import OpenAI, RateLimitError, APIError
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DEFAULT_MODEL = "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ"
VLLM_BASE_URL = "http://localhost:8081/v1"
DEFAULT_OUTPUT_PATH = "genz_dataset.jsonl"
DEFAULT_LIMIT = 500
OPIK_PROJECT_NAME = "genz-speak-data-gen"

GENZ_SYSTEM_PROMPT = (
    "You are a Gen Z teenager who speaks in current internet slang. "
    "You MUST keep all factual information 100% accurate — only the style changes."
)

GENZ_USER_PROMPT_TEMPLATE = (
    "Rewrite the following answer to sound exactly like a Gen Z teenager wrote it. "
    "Keep ALL the facts, just change the vibe. "
    "Do NOT add disclaimers or meta-comments. Output ONLY the rewritten answer.\n\n"
    "Original answer:\n{answer}"
)



@dataclass
class VLLMClient:

    model: str = DEFAULT_MODEL
    base_url: str = VLLM_BASE_URL
    max_retries: int = 5
    retry_base_delay: float = 2.0
    temperature: float = 0.85
    max_tokens: int = 1024

    _client: OpenAI = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = OpenAI(
            base_url=self.base_url,
            api_key="EMPTY",
        )

    @opik.track(name="vllm-chat", project_name=OPIK_PROJECT_NAME)
    def chat(
        self,
        system: str,
        user: str,
    ) -> str:

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("vLLM returned an empty response.")
                return content.strip()

            except RateLimitError as exc:
                wait = self.retry_base_delay * (2 ** (attempt - 1))
                log.warning(
                    "Rate-limited (attempt %d/%d). Retrying in %.1fs. %s",
                    attempt,
                    self.max_retries,
                    wait,
                    exc,
                )
                if attempt == self.max_retries:
                    raise
                time.sleep(wait)

            except APIError as exc:
                log.error("vLLM API error on attempt %d: %s", attempt, exc)
                if attempt == self.max_retries:
                    raise
                time.sleep(self.retry_base_delay)

        raise RuntimeError("Exhausted retries without success.")



@opik.track(name="load-dolly-open-qa", project_name=OPIK_PROJECT_NAME)
def load_dolly_open_qa(limit: int) -> list[dict]:
    log.info("Loading databricks-dolly-15k (open_qa) …")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    open_qa = [row for row in dataset if row["category"] == "open_qa"]
    log.info("Found %d open_qa examples (capping at %d).", len(open_qa), limit)
    return open_qa[:limit]


def _stream_genz_rewrites(
    rows: list[dict],
    client: VLLMClient,
) -> Iterator[dict]:
    for row in rows:
        instruction: str = row.get("instruction", "").strip()
        original_answer: str = (
            row.get("response") or row.get("output") or ""
        ).strip()

        if not instruction or not original_answer:
            log.debug("Skipping row with empty instruction/answer.")
            continue

        user_prompt = GENZ_USER_PROMPT_TEMPLATE.format(answer=original_answer)

        try:
            genz_answer = client.chat(
                system=GENZ_SYSTEM_PROMPT,
                user=user_prompt,
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failed to rewrite row (instruction: %.60s…): %s",
                instruction,
                exc,
            )
            continue

        yield {"instruction": instruction, "output": genz_answer}


@opik.track(name="generate-genz-dataset", project_name=OPIK_PROJECT_NAME)
def generate_dataset(
    rows: list[dict],
    client: VLLMClient,
    output_path: Path,
) -> int:
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with output_path.open("w", encoding="utf-8") as fh:
        for record in tqdm(
            _stream_genz_rewrites(rows, client),
            total=len(rows),
            desc="Rewriting in Gen Z",
            unit="row",
        ):
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    log.info("Saved %d Gen Z examples to %s", written, output_path)
    return written




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Gen Z–style fine-tuning dataset from dolly-15k."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Max number of open_qa rows to process (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"vLLM model ID served at --base-url (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=VLLM_BASE_URL,
        help=f"vLLM OpenAI-compatible base URL (default: {VLLM_BASE_URL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT_PATH),
        help=f"Output JSONL file path (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    opik.configure(use_local=False)

    client = VLLMClient(model=args.model, base_url=args.base_url)
    rows = load_dolly_open_qa(limit=args.limit)
    n_written = generate_dataset(rows=rows, client=client, output_path=args.output)

    log.info("Done. %d Gen Z examples written to '%s'.", n_written, args.output)


if __name__ == "__main__":
    main()
