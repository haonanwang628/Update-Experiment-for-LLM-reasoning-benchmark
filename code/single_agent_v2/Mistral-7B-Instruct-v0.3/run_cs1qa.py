"""CS1QA single-agent evaluation with a free-text reasoning prompt and an independent LLM judge.

Python 3.10+, requests. Talks to two OpenAI-compatible chat completions endpoints:
one for the model under test (free-text reasoning prompt, answer taken from the
"Final answer:" line) and one for an independent judge model (returns exactly
CORRECT or INCORRECT per sample). No model weights or torch are required here;
both endpoints are expected to already be serving (e.g. via vLLM).
"""

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import requests

ROOT = Path(__file__).resolve().parent
# Some models emit "Final answer: ..." on one line; others emit a markdown
# heading like "### Final answer" with the answer starting on the next line.
FINAL_ANSWER_RE = re.compile(r"#{0,6}\s*final answer\s*:?\s*(.*)", re.IGNORECASE | re.DOTALL)
INCORRECT_RE = re.compile(r"\bINCORRECT\b", re.IGNORECASE)
CORRECT_RE = re.compile(r"\bCORRECT\b", re.IGNORECASE)


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fill(template, mapping):
    text = template
    for key, value in mapping.items():
        text = text.replace("{{" + key + "}}", value)
    if re.search(r"\{\{.*?\}\}", text):
        raise ValueError("Unresolved placeholder in prompt")
    return text


def extract_final_answer(text):
    matches = list(FINAL_ANSWER_RE.finditer(text))
    if not matches:
        raise ValueError("No 'Final answer:' section found in model output")
    answer = matches[-1].group(1).strip()
    if not answer:
        raise ValueError("Empty final answer")
    return answer


def extract_verdict(text):
    stripped = text.strip()
    # Check INCORRECT first: "CORRECT" is a substring of "INCORRECT".
    if INCORRECT_RE.search(stripped):
        return False
    if CORRECT_RE.search(stripped):
        return True
    raise ValueError("Judge output was neither CORRECT nor INCORRECT: " + stripped[:200])


class ChatClient:
    def __init__(self, base_url, key, timeout, retries):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.key, self.timeout, self.retries = key, timeout, retries

    def complete(self, model, messages, max_tokens, temperature, top_p=0.9, extra_body=None):
        payload = {"model": model, "messages": messages,
                   "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p, "stream": False}
        if extra_body:
            payload.update(extra_body)
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(self.url, headers=headers, json=payload, timeout=(15, self.timeout))
                if response.status_code >= 400:
                    retryable = response.status_code in (408, 429) or response.status_code >= 500
                    if retryable and attempt < self.retries:
                        time.sleep(min(2 ** attempt, 30))
                        continue
                    detail = ""
                    try:
                        error = response.json().get("error", {})
                        if isinstance(error, dict) and isinstance(error.get("message"), str):
                            detail = error["message"]
                            if self.key:
                                detail = detail.replace(self.key, "[REDACTED]")
                            detail = re.sub(r"(?i)Bearer\s+\S+", "Bearer [REDACTED]", detail)
                            detail = " ".join(detail.split())[:800]
                    except (AttributeError, TypeError, ValueError):
                        pass
                    raise RuntimeError("API HTTP {}{}".format(
                        response.status_code, ": " + detail if detail else ""))
                data = response.json()
                choice = data["choices"][0]
                content = choice["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError("API response content is not text")
                return {"text": content, "finish_reason": choice.get("finish_reason"),
                        "usage": data.get("usage"), "returned_model": data.get("model")}
            except (requests.Timeout, requests.ConnectionError):
                if attempt == self.retries:
                    raise RuntimeError("API connection error or timeout") from None
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError("API retries exhausted")


def evaluate_one(row, index, args, reasoning_template, judge_template, client, judge):
    qid = "cs1qa:test:{}".format(index)
    question = row["question"]
    code = row.get("code") or ""
    reference = row["answer"]
    result = {"index": index, "question_id": qid, "reference": reference,
              "status": "api_error", "correct": None,
              "labNo": row.get("labNo"), "taskNo": row.get("taskNo"), "questionType": row.get("questionType")}
    started = time.monotonic()
    try:
        prompt = fill(reasoning_template, {"question": question, "code": code})
        result["response"] = client.complete(args.model, [{"role": "user", "content": prompt}],
                                               args.max_tokens, args.temperature, args.top_p)
        result["status"] = "parse_error"
        if result["response"]["finish_reason"] == "length":
            raise ValueError("Model output truncated at max_tokens")
        answer = extract_final_answer(result["response"]["text"])
        result["candidate_answer"] = answer

        result["status"] = "judge_error"
        judge_prompt = fill(judge_template, {"question": question, "code": code,
                                              "reference_answer": reference, "candidate_answer": answer})
        result["judge_response"] = judge.complete(args.judge_model, [{"role": "user", "content": judge_prompt}],
                                                    args.judge_max_tokens, 0, 1.0,
                                                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        if result["judge_response"]["finish_reason"] == "length":
            raise ValueError("Judge output truncated at max_tokens")
        result["correct"] = extract_verdict(result["judge_response"]["text"])
        result["status"] = "ok"
    except Exception as exc:
        result["error"] = "{}: {}".format(type(exc).__name__, exc)
    result["seconds"] = round(time.monotonic() - started, 3)
    return result


def token_usage(results, response_key):
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    totals, reported = Counter(), Counter()
    responses = 0
    for record in results:
        for attempt in record.get("previous_attempts", []) + [record]:
            response = attempt.get(response_key)
            if not isinstance(response, dict):
                continue
            responses += 1
            usage = response.get("usage") or {}
            if not isinstance(usage, dict):
                usage = {}
            values = {k: usage.get(k) for k in fields}
            valid = lambda value: type(value) is int and value >= 0
            if not valid(values["total_tokens"]) and all(valid(values[k]) for k in fields[:2]):
                values["total_tokens"] = values["prompt_tokens"] + values["completion_tokens"]
            for field, value in values.items():
                if valid(value):
                    totals[field] += value
                    reported[field] += 1
    return {**{k: totals[k] if reported[k] else None for k in fields},
            "responses": responses,
            "missing_usage": {k: responses - reported[k] for k in fields},
            "all_saved_responses_reported_usage": responses > 0 and all(reported[k] == responses for k in fields)}


def summarize(results, total):
    statuses = Counter(x["status"] for x in results)
    scored = [x for x in results if x["correct"] is not None]
    correct = sum(x["correct"] is True for x in scored)
    by_type = {}
    for qtype in sorted({x["questionType"] for x in results if x.get("questionType")}):
        subset = [x for x in results if x.get("questionType") == qtype and x["correct"] is not None]
        by_type[qtype] = {"samples": sum(x.get("questionType") == qtype for x in results),
                           "scored": len(subset),
                           "accuracy_on_scored": sum(x["correct"] is True for x in subset) / len(subset) if subset else None}
    return {"dataset": "cs1qa", "split": "test", "selected_samples": total, "saved_samples": len(results),
            "statuses": dict(statuses),
            "model_token_usage": token_usage(results, "response"),
            "judge_token_usage": token_usage(results, "judge_response"),
            "metric": "judge_accuracy",
            "correct": correct, "scored_samples": len(scored),
            "accuracy": correct / total if total else None,
            "accuracy_on_scored": correct / len(scored) if scored else None,
            "complete": len(results) == total and statuses.get("ok", 0) == total,
            "by_question_type": by_type,
            "note": "Accuracy uses all selected samples as denominator; errors/missing/unparseable outputs count as incorrect."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "test_cleaned.jsonl")
    parser.add_argument("--reasoning-prompt", type=Path, default=ROOT / "single-agent-CS1QA-reasoningprompt.txt")
    parser.add_argument("--judge-prompt", type=Path, default=ROOT / "single-agent-CS1QA-judgeprompt.txt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--model", required=True, help="Model name as registered with the serving endpoint")
    parser.add_argument("--base-url", default=os.getenv("MODEL_BASE_URL"), help="OpenAI-compatible base URL, including /v1")
    parser.add_argument("--api-key-env", default="MODEL_API_KEY")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-base-url", default=os.getenv("JUDGE_BASE_URL"))
    parser.add_argument("--judge-key-env", default="JUDGE_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--judge-max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, help="First N samples; use a separate --output-dir")
    parser.add_argument("--dry-run", action="store_true", help="Print one resolved prompt pair, no API calls")
    parser.add_argument("--score-only", action="store_true", help="Aggregate saved records without calling any API")
    parser.add_argument("--retry-errors", action="store_true", help="On resume, repeat failed requests")
    args = parser.parse_args()

    if not 0 <= args.temperature <= 2 or not 0 < args.top_p <= 1:
        parser.error("temperature must be in [0, 2]; top-p must be in (0, 1]")
    if args.concurrency < 1 or args.max_tokens < 1 or args.judge_max_tokens < 1 or args.timeout <= 0 or args.retries < 0:
        parser.error("Counts and timeouts must be positive; retries must be nonnegative")
    if args.judge_model == args.model and (args.judge_base_url or args.base_url) == args.base_url:
        parser.error("Use an independent judge model and/or endpoint, not the model being evaluated")

    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[:args.limit]
    reasoning_template = args.reasoning_prompt.read_text(encoding="utf-8-sig")
    judge_template = args.judge_prompt.read_text(encoding="utf-8-sig")

    if args.dry_run:
        row = rows[0]
        prompt = fill(reasoning_template, {"question": row["question"], "code": row.get("code") or ""})
        judge_prompt = fill(judge_template, {"question": row["question"], "code": row.get("code") or "",
                                              "reference_answer": row["answer"], "candidate_answer": "<example candidate answer>"})
        print("=== samples:", len(rows), "===")
        print("=== reasoning prompt (sample 0) ===\n" + prompt)
        print("=== judge prompt (sample 0) ===\n" + judge_prompt)
        return 0

    if not (args.score_only) and not args.base_url:
        parser.error("Set MODEL_BASE_URL or --base-url (including /v1)")

    config = {"model": args.model, "base_url": args.base_url,
              "judge_model": args.judge_model, "judge_base_url": args.judge_base_url or args.base_url,
              "data_sha256": digest(args.data.read_bytes()),
              "reasoning_prompt_sha256": digest(reasoning_template.encode()),
              "judge_prompt_sha256": digest(judge_template.encode()),
              "max_tokens": args.max_tokens, "judge_max_tokens": args.judge_max_tokens,
              "temperature": args.temperature, "top_p": args.top_p, "limit": args.limit,
              "runner_sha256": digest(Path(__file__).read_bytes())}
    run_path = args.output_dir / "run.json"
    if run_path.exists():
        if read_json(run_path) != config:
            parser.error("Run configuration changed. Use the original arguments or a new --output-dir.")
    elif args.score_only:
        parser.error("No run.json found for --score-only")
    else:
        save_json(run_path, config)

    client = ChatClient(args.base_url or "", os.getenv(args.api_key_env, ""), args.timeout, args.retries)
    judge_key = os.getenv(args.judge_key_env, "") or os.getenv(args.api_key_env, "")
    judge = ChatClient(args.judge_base_url or args.base_url or "", judge_key, args.timeout, args.retries)

    results, pending = {}, []
    for index, row in enumerate(rows):
        path = args.output_dir / "records" / ("{:06d}.json".format(index))
        if path.exists():
            saved = read_json(path)
            results[index] = saved
            if not args.retry_errors or saved["status"] == "ok":
                continue
        pending.append((index, row))

    if not args.score_only and pending:
        pool = ThreadPoolExecutor(max_workers=args.concurrency)
        iterator = iter(pending)
        active = set()
        try:
            def enqueue():
                item = next(iterator, None)
                if item is not None:
                    index, row = item
                    active.add(pool.submit(evaluate_one, row, index, args, reasoning_template, judge_template, client, judge))
            for _ in range(args.concurrency):
                enqueue()
            while active:
                done, active = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    previous = results.get(result["index"])
                    if previous is not None:
                        result["previous_attempts"] = previous.get("previous_attempts", []) + [
                            {k: v for k, v in previous.items() if k != "previous_attempts"}]
                    save_json(args.output_dir / "records" / ("{:06d}.json".format(result["index"])), result)
                    results[result["index"]] = result
                    print("{}/{} {}".format(len(results), len(rows), result["status"]), flush=True)
                    if result.get("error"):
                        print("  {}: {}".format(result["question_id"], result["error"]), file=sys.stderr, flush=True)
                    enqueue()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    ordered = [results[i] for i in sorted(results)]
    metrics = summarize(ordered, len(rows))
    save_json(args.output_dir / "predictions.json", ordered)
    save_json(args.output_dir / "metrics.json", metrics)
    print(json.dumps({"dataset": "cs1qa", "metric": metrics["metric"], "accuracy": metrics["accuracy"],
                       "accuracy_on_scored": metrics["accuracy_on_scored"], "complete": metrics["complete"]}))
    return 1 if not metrics["complete"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted. Re-run the same command to resume from saved records.", file=sys.stderr)
        sys.exit(130)
