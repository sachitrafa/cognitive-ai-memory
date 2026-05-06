"""
FEVER Contradiction Handling Benchmark
---------------------------------------
Tests whether YourMemory correctly resolves contradicting facts using the FEVER
dataset (Fact Extraction and VERification, Thorne et al. 2018).

FEVER has 185,445 claims derived from Wikipedia, each labelled:
  SUPPORTS   — claim is supported by the evidence
  REFUTES    — claim is refuted by the evidence (contradicts a true fact)
  NOT ENOUGH INFO

Benchmark design:
  1. For each SUPPORTS/REFUTES pair on the same subject:
       - Store the SUPPORTS fact (the true fact)
       - Store the REFUTES claim (the contradicting false fact)
  2. Query with the original question
  3. Score:
       RESOLVED  — retrieved memory is the SUPPORTS fact (correct)
       CORRUPTED — retrieved memory is the REFUTES claim (bad)
       CONFUSED  — both returned (no contradiction resolution)

Download the dataset:
    wget https://fever.ai/download/fever/train.jsonl -O ~/fever_train.jsonl
    wget https://fever.ai/download/fever/shared_task_dev.jsonl -O ~/fever_dev.jsonl

Usage:
    python benchmarks/fever_contradiction.py
    python benchmarks/fever_contradiction.py --limit 200 --data ~/fever_dev.jsonl
    python benchmarks/fever_contradiction.py --limit 200 --split dev
"""

import sys, os, json, argparse, time, re
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("YOURMEMORY_DB", os.path.expanduser("~/.yourmemory/benchmark_fever.duckdb"))

from src.services.embed import embed as _embed_raw
from src.services.retrieve import retrieve
from src.routes.memories import add_memory, MemoryRequest

_EMB_CACHE: dict[str, list] = {}

def embed(text: str) -> list:
    if text not in _EMB_CACHE:
        _EMB_CACHE[text] = _embed_raw(text)
    return _EMB_CACHE[text]


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_DEV  = os.path.expanduser("~/fever_dev.jsonl")
DEFAULT_WIKI = os.path.expanduser("~/fever_wiki_pages/")   # optional, for evidence text
TOP_K        = 5
SIM_THRESH   = 0.25
USER_ID      = "fever_bench"


# ── Helpers ───────────────────────────────────────────────────────────────────

def cosine(a, b):
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def _primary_page(evidence: list) -> str | None:
    """Extract the main Wikipedia page from a FEVER evidence list."""
    for group in evidence:
        for ev in group:
            if len(ev) >= 3 and ev[2]:
                return ev[2]
    return None


def load_fever_pairs(path: str, limit: int) -> list[dict]:
    """
    Load SUPPORTS/REFUTES pairs from FEVER jsonl.
    Pairs by the primary Wikipedia evidence page — the correct way to find
    claims that are genuinely about the same subject.

    Returns list of {true_claim, false_claim, query} dicts.
    """
    supports: dict[str, str] = {}   # page → claim
    refutes:  dict[str, str] = {}

    with open(path) as f:
        for line in f:
            item = json.loads(line)
            label = item.get("label", "")
            claim = item.get("claim", "").strip()
            evidence = item.get("evidence", [])
            if not claim or not evidence:
                continue
            page = _primary_page(evidence)
            if not page:
                continue
            if label == "SUPPORTS" and page not in supports:
                supports[page] = claim
            elif label == "REFUTES" and page not in refutes:
                refutes[page] = claim

    pairs = []
    for page in supports:
        if page in refutes:
            true_claim  = supports[page]
            false_claim = refutes[page]
            pairs.append({
                "true_claim":  true_claim,
                "false_claim": false_claim,
                "query":       true_claim,
            })
            if len(pairs) >= limit * 2:
                break

    return pairs[:limit]


def _safe_store(content: str) -> dict:
    try:
        return add_memory(MemoryRequest(userId=USER_ID, content=content, importance=0.8))
    except Exception:
        return {}


def store_pair(true_claim: str, false_claim: str, pair_id: int):
    """Store true fact first, then contradicting false claim. Returns (true_id, false_id)."""
    r1 = _safe_store(true_claim)
    time.sleep(0.05)   # ensure distinct timestamps
    r2 = _safe_store(false_claim)
    return r1.get("id"), r2.get("id")


def score_result(memories: list, true_claim: str, false_claim: str) -> str:
    """
    Score retrieval result as RESOLVED / CORRUPTED / CONFUSED / MISSED.
    """
    if not memories:
        return "MISSED"

    true_emb  = embed(true_claim)
    false_emb = embed(false_claim)

    found_true  = False
    found_false = False

    for m in memories[:TOP_K]:
        mem_emb = embed(m["content"])
        sim_true  = cosine(mem_emb, true_emb)
        sim_false = cosine(mem_emb, false_emb)
        if sim_true >= 0.80:
            found_true = True
        if sim_false >= 0.80:
            found_false = True

    if found_true and not found_false:
        return "RESOLVED"
    if found_false and not found_true:
        return "CORRUPTED"
    if found_true and found_false:
        return "CONFUSED"
    return "MISSED"


# ── Main ──────────────────────────────────────────────────────────────────────

def run(data_path: str, limit: int, verbose: bool):
    print(f"\nFEVER Contradiction Benchmark  |  limit={limit}  |  {data_path}")
    print("=" * 65)

    if not os.path.exists(data_path):
        print(f"\nDataset not found: {data_path}")
        print("Download with:")
        print("  wget https://fever.ai/download/fever/shared_task_dev.jsonl -O ~/fever_dev.jsonl")
        sys.exit(1)

    print("Loading FEVER pairs...", end=" ", flush=True)
    pairs = load_fever_pairs(data_path, limit)
    print(f"{len(pairs)} pairs loaded")

    results = {"RESOLVED": 0, "CORRUPTED": 0, "CONFUSED": 0, "MISSED": 0}
    t0 = time.time()

    for i, pair in enumerate(pairs):
        true_id, false_id = store_pair(pair["true_claim"], pair["false_claim"], i)

        result = retrieve(USER_ID, pair["query"], top_k=TOP_K)
        memories = result.get("memories", [])

        outcome = score_result(memories, pair["true_claim"], pair["false_claim"])
        results[outcome] += 1

        if verbose:
            print(f"  [{i+1:3d}/{len(pairs)}] {outcome:10s} | {pair['true_claim'][:60]}")
        elif (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(pairs)} done  ({elapsed:.0f}s)  "
                  f"resolved={results['RESOLVED']}  corrupted={results['CORRUPTED']}")

    elapsed = time.time() - t0
    total = len(pairs)

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"  RESOLVED   (correct fact surfaced)   : {results['RESOLVED']:4d} / {total}  "
          f"({results['RESOLVED']/total*100:.1f}%)")
    print(f"  CORRUPTED  (false claim surfaced)     : {results['CORRUPTED']:4d} / {total}  "
          f"({results['CORRUPTED']/total*100:.1f}%)")
    print(f"  CONFUSED   (both surfaced)            : {results['CONFUSED']:4d} / {total}  "
          f"({results['CONFUSED']/total*100:.1f}%)")
    print(f"  MISSED     (neither surfaced)         : {results['MISSED']:4d} / {total}  "
          f"({results['MISSED']/total*100:.1f}%)")
    print(f"\n  Contradiction Resolution Rate         : {results['RESOLVED']/total*100:.1f}%")
    print(f"  Total time: {elapsed:.1f}s")

    # Save results
    os.makedirs("benchmarks/results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "benchmark": "FEVER Contradiction Handling",
        "dataset":   data_path,
        "limit":     limit,
        "timestamp": ts,
        "results":   results,
        "contradiction_resolution_rate": round(results["RESOLVED"] / total, 4),
        "corruption_rate":               round(results["CORRUPTED"] / total, 4),
    }
    out_path = f"benchmarks/results/fever_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default=DEFAULT_DEV)
    parser.add_argument("--limit",   type=int, default=500)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(args.data, args.limit, args.verbose)
