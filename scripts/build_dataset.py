import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_URL = (
    "https://huggingface.co/datasets/novita/agentic_code_dataset_22/resolve/main/"
    "e22_sessions_openai.json"
)
CACHE_PATH = REPO_ROOT / ".cache" / "datasets" / "e22_sessions_openai.json"
CHARS_PER_TOKEN = 3.5


def download() -> Path:
    if CACHE_PATH.exists() and CACHE_PATH.stat().st_size > 1_000_000_000:
        return CACHE_PATH
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".part")
    print(f"downloading {DATASET_URL}")
    with urllib.request.urlopen(DATASET_URL) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(1 << 24)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB ({100 * done // total}%)",
                      end="", flush=True)
    print()
    tmp.rename(CACHE_PATH)
    return CACHE_PATH


def load_sessions(path: Path) -> list[list[dict]]:
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        for key in ("sessions", "data", "conversations"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
        else:
            raw = list(raw.values())
    sessions = []
    for entry in raw:
        messages = entry.get("messages") or entry.get("turns") or entry.get("conversation") if isinstance(entry, dict) else None
        if not isinstance(messages, list):
            continue
        turns = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role not in ("user", "assistant") or not isinstance(content, str):
                continue
            content = content.strip()
            if content:
                turns.append({"role": role, "content": content})
        if len(turns) >= 2:
            sessions.append(turns)
    if not sessions:
        print("FATAL: no usable sessions found in the corpus", file=sys.stderr)
        sys.exit(1)
    return sessions


def est_tokens(turns: list[dict]) -> int:
    return int(sum(len(t["content"]) for t in turns) / CHARS_PER_TOKEN)


def sample_windows(sessions: list[list[dict]], n_samples: int, min_tokens: int,
                   max_tokens: int, seed: int) -> list[dict]:
    windows: list[dict] = []
    for sid, turns in enumerate(sessions):
        start = 0
        while start < len(turns) - 1:
            end = start + 1
            while end < len(turns) and est_tokens(turns[start:end]) < min_tokens:
                end += 1
            if end > len(turns) - 1:
                break
            while end < len(turns) and est_tokens(turns[start:end + 1]) <= max_tokens:
                end += 1
            window = turns[start:end]
            tokens = est_tokens(window)
            if min_tokens <= tokens <= max_tokens:
                windows.append({"session": sid, "start": start, "turns": window, "tokens": tokens})
            start = end
    if not windows:
        print("FATAL: no conversation windows matched the token range; "
              "widen --min-tokens/--max-tokens", file=sys.stderr)
        sys.exit(1)

    pools: dict[int, list[dict]] = {}
    for w in windows:
        pools.setdefault(w["session"], []).append(w)
    for pool in pools.values():
        pool.sort(key=lambda w: hashlib.sha256(
            f"{seed}:{w['session']}:{w['start']}".encode()).hexdigest())
    ordered = sorted(pools, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())

    samples, idx = [], 0
    while len(samples) < n_samples and any(pools[s] for s in ordered):
        sid = ordered[idx % len(ordered)]
        idx += 1
        if pools[sid]:
            samples.append(pools[sid].pop(0))
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--min-tokens", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output",
                    default=str(REPO_ROOT / "benchmarks" / "datasets" / "agentic_code22.jsonl"))
    args = ap.parse_args()

    path = download()
    sessions = load_sessions(path)
    print(f"{len(sessions)} sessions, {sum(len(s) for s in sessions)} usable turns")

    samples = sample_windows(sessions, args.samples, args.min_tokens, args.max_tokens, args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i, w in enumerate(samples):
            f.write(json.dumps({
                "conversation_id": i,
                "turns": w["turns"],
                "session_id": w["session"],
                "window_start_turn": w["start"],
                "est_input_tokens": w["tokens"],
            }) + "\n")

    tokens = sorted(w["tokens"] for w in samples)
    stats = {
        "source": "novita/agentic_code_dataset_22",
        "source_file": CACHE_PATH.name,
        "samples": len(samples),
        "sessions_covered": len({w["session"] for w in samples}),
        "seed": args.seed,
        "token_range": [args.min_tokens, args.max_tokens],
        "est_input_tokens": {
            "min": tokens[0],
            "p50": tokens[len(tokens) // 2],
            "max": tokens[-1],
            "mean": sum(tokens) // len(tokens),
        },
        "turns_per_sample": {
            "min": min(len(w["turns"]) for w in samples),
            "mean": sum(len(w["turns"]) for w in samples) // len(samples),
            "max": max(len(w["turns"]) for w in samples),
        },
    }
    (out_path.parent / "dataset-stats.json").write_text(json.dumps(stats, indent=2))
    print(f"wrote {len(samples)} samples to {out_path}")
    print(f"  est. input tokens: p50={stats['est_input_tokens']['p50']} "
          f"mean={stats['est_input_tokens']['mean']} max={stats['est_input_tokens']['max']}")


if __name__ == "__main__":
    main()
