"""GSM8K single-agent eval for the V2 plain-text prompt
(single-agent-GSM8K copy.txt): one user message per question with
{{question}} substituted in, scored by extracting the "Final answer:"
line.

Model: microsoft/Phi-4-mini-instruct via vLLM (served as
phi-4-mini-instruct), temperature 0.3, top_p 0.9, max_tokens 2048 — same
generation config and scoring methodology as the Mistral-7B-Instruct-v0.3
V2 runs, against the same local GSM8K data/gold answers used before.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

SINGLE_AGENT_DIR = next(
    p / "single-agent" for p in Path(__file__).resolve().parents
    if (p / "single-agent" / "run_single.py").exists()
)
sys.path.insert(0, str(SINGLE_AGENT_DIR))
from run_single import number, read_json, save_json  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA_FILE = SINGLE_AGENT_DIR / "data" / "gsm8k" / "test.json"

FINAL_ANSWER = re.compile(r'final\s*answer\s*:\s*(.+)', re.IGNORECASE)
NUMBER_TOKEN = re.compile(r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?')


def last_number_in(text):
    matches = NUMBER_TOKEN.findall(text)
    return number(matches[-1]) if matches else None


def extract_final_answer(raw_text):
    """Return (value: Decimal|None, method: str)."""
    matches = FINAL_ANSWER.findall(raw_text)
    if matches:
        candidate = matches[-1].strip()
        direct = number(candidate)
        if direct is not None:
            return direct, "final_answer_line_direct"
        salvaged = last_number_in(candidate)
        if salvaged is not None:
            return salvaged, "final_answer_line_last_number"
    salvaged = last_number_in(raw_text)
    if salvaged is not None:
        return salvaged, "raw_text_last_number"
    return None, "no_number_found"


def format_ok(raw_text):
    """All four required field labels present, in order."""
    labels = ["reasoning", "initial answer", "reflection", "final answer"]
    lowered = raw_text.lower()
    positions = [lowered.find(l) for l in labels]
    return all(p != -1 for p in positions) and positions == sorted(positions)


class ChatClient:
    def __init__(self, base_url, timeout, retries):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.timeout, self.retries = timeout, retries

    def complete(self, model, messages, max_tokens, temperature, top_p):
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature, "top_p": top_p, "stream": False}
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(self.url, headers={"Content-Type": "application/json"},
                                          json=payload, timeout=(15, self.timeout))
                if response.status_code >= 400:
                    retryable = response.status_code in (408, 429) or response.status_code >= 500
                    if retryable and attempt < self.retries:
                        time.sleep(min(2 ** attempt, 30))
                        continue
                    raise RuntimeError("API HTTP {}".format(response.status_code))
                data = response.json()
                choice = data["choices"][0]
                return {"text": choice["message"]["content"], "finish_reason": choice.get("finish_reason"),
                        "usage": data.get("usage"), "returned_model": data.get("model")}
            except (requests.Timeout, requests.ConnectionError):
                if attempt == self.retries:
                    raise RuntimeError("API connection error or timeout") from None
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError("API retries exhausted")


def evaluate_one(index, row, template, gold, args, client):
    result = {"index": index, "status": "api_error", "correct": False}
    started = time.monotonic()
    try:
        prompt = template.replace("{{question}}", row["question"])
        if "{{question}}" in prompt:
            raise ValueError("Unresolved prompt placeholder")
        result["response"] = client.complete(
            args.model, [{"role": "user", "content": prompt}],
            args.max_tokens, args.temperature, args.top_p)
        result["status"] = "scored"
        raw_text = result["response"]["text"]
        result["format_ok"] = format_ok(raw_text)
        value, method = extract_final_answer(raw_text)
        result["extraction_method"] = method
        result["extracted_answer"] = str(value) if value is not None else None
        result["gold_answer"] = str(gold)
        result["correct"] = value is not None and value == gold
    except Exception as exc:
        result["error"] = "{}: {}".format(type(exc).__name__, exc)
    result["seconds"] = round(time.monotonic() - started, 3)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", type=Path, default=ROOT / "single-agent-GSM8K copy.txt")
    ap.add_argument("--data-file", type=Path, default=DATA_FILE)
    ap.add_argument("--output-dir", type=Path, default=ROOT)
    ap.add_argument("--model", default="phi-4-mini-instruct")
    ap.add_argument("--base-url", default=os.getenv("MODEL_BASE_URL", "http://localhost:8001/v1"))
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    template = args.prompt.read_text(encoding="utf-8-sig")
    rows = read_json(args.data_file)
    if args.limit:
        rows = rows[:args.limit]
    gold = [number(row["answer"].rsplit("####", 1)[-1].strip()) for row in rows]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prompt.txt").write_text(template, encoding="utf-8")
    records_dir = args.output_dir / "records"
    records_dir.mkdir(exist_ok=True)

    config = {"model": args.model, "base_url": args.base_url, "dataset": "gsm8k", "split": "test",
              "samples": len(rows), "max_tokens": args.max_tokens, "temperature": args.temperature,
              "top_p": args.top_p, "prompt_file": str(args.prompt)}
    save_json(args.output_dir / "run.json", config)

    client = ChatClient(args.base_url, args.timeout, args.retries)
    results = {}
    pending = []
    for index in range(len(rows)):
        path = records_dir / "{:06d}.json".format(index)
        if path.exists():
            results[index] = json.loads(path.read_text(encoding="utf-8"))
            continue
        pending.append(index)

    if pending:
        pool = ThreadPoolExecutor(max_workers=args.concurrency)
        iterator = iter(pending)
        active = set()
        try:
            def enqueue():
                idx = next(iterator, None)
                if idx is not None:
                    active.add(pool.submit(evaluate_one, idx, rows[idx], template, gold[idx], args, client))
            for _ in range(args.concurrency):
                enqueue()
            while active:
                done, active = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    save_json(records_dir / "{:06d}.json".format(result["index"]), result)
                    results[result["index"]] = result
                    print("gsm8k_v2 {}/{} {}".format(len(results), len(rows), result["status"]), flush=True)
                    if result.get("error"):
                        print("  index {}: {}".format(result["index"], result["error"]), file=sys.stderr, flush=True)
                    enqueue()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    ordered = [results[i] for i in sorted(results)]
    correct = sum(r.get("correct") is True for r in ordered)
    format_compliant = sum(r.get("format_ok") is True for r in ordered)
    scored = sum(r.get("status") == "scored" for r in ordered)
    unextractable = sum(r.get("extraction_method") == "no_number_found" for r in ordered)
    methods = {}
    for r in ordered:
        m = r.get("extraction_method")
        if m:
            methods[m] = methods.get(m, 0) + 1

    summary = {
        "dataset": "gsm8k", "split": "test", "prompt_version": "V2_plain_text",
        "total_samples": len(rows), "scored_samples": scored,
        "correct": correct, "accuracy": correct / len(rows),
        "format_compliant": format_compliant, "format_compliant_rate": format_compliant / len(rows),
        "unextractable_no_number_found": unextractable,
        "extraction_method_counts": methods,
        "model_config": {"model": args.model, "base_url": args.base_url, "temperature": args.temperature,
                          "top_p": args.top_p, "max_tokens": args.max_tokens},
        "note": ("Accuracy counts a sample correct if the Final answer line's numeric value "
                 "(robustly extracted; text scanning only used when the exact 'Final answer:' "
                 "line is missing or non-numeric) matches the gold GSM8K answer. "
                 "format_compliant / format_compliant_rate report how many responses contained "
                 "all four required field labels (Reasoning/Initial answer/Reflection/Final answer) "
                 "in order, kept separate from correctness."),
    }
    save_json(args.output_dir / "metrics.json", summary)
    save_json(args.output_dir / "predictions.json", ordered)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
