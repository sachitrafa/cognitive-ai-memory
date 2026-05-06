"""
HotpotQA Multi-Hop Reasoning Benchmark
---------------------------------------
Tests whether YourMemory can surface both supporting facts needed to answer
multi-hop questions from the HotpotQA dataset (Yang et al. 2018).

HotpotQA has 113,000 crowd-sourced QA pairs that REQUIRE combining two
supporting passages — unlike single-hop QA, no single fact is sufficient.
This exercises graph expansion (Round 2 retrieval) directly: the first
supporting fact is a Round 1 vector match; the second is a graph neighbour.

Scoring:
  BOTH_FOUND   — both supporting facts retrieved (graph expansion working)
  ONE_FOUND    — only first supporting fact found (graph expansion missed)
  NONE_FOUND   — neither found (retrieval failure)

Download the dataset:
    wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json \
         -O ~/hotpot_dev.json
    # Or distractor set (smaller, 7,405 questions):
    wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json \
         -O ~/hotpot_dev_distractor.json

Usage:
    python benchmarks/hotpotqa_reasoning.py
    python benchmarks/hotpotqa_reasoning.py --limit 200
    python benchmarks/hotpotqa_reasoning.py --data ~/hotpot_dev_distractor.json --limit 300
    python benchmarks/hotpotqa_reasoning.py --verbose
"""

import sys, os, json, argparse, time
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("YOURMEMORY_DB", os.path.expanduser("~/.yourmemory/benchmark_hotpot.duckdb"))

from src.services.embed import embed as _embed_raw
from src.services.retrieve import retrieve
from src.routes.memories import add_memory, MemoryRequest

_EMB_CACHE: dict[str, list] = {}

def embed(text: str) -> list:
    if text not in _EMB_CACHE:
        _EMB_CACHE[text] = _embed_raw(text)
    return _EMB_CACHE[text]


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_DATA = os.path.expanduser("~/hotpot_dev_distractor.json")
TOP_K        = 5
USER_ID      = "hotpot_bench"
SIM_HIT      = 0.75   # cosine threshold to count a supporting fact as "found"


# ── Helpers ───────────────────────────────────────────────────────────────────

def cosine(a, b):
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def first_sentence(text: str) -> str:
    """Extract first sentence (or first 200 chars) as the supporting fact."""
    for sep in (". ", ".\n", "! ", "? "):
        idx = text.find(sep)
        if 40 <= idx <= 300:
            return text[:idx + 1].strip()
    return text[:200].strip()


def load_hotpot(path: str, limit: int) -> list[dict]:
    """
    Load HotpotQA examples with exactly two supporting facts.
    Returns list of {question, answer, fact1, fact2, type} dicts.
    """
    with open(path) as f:
        data = json.load(f)

    examples = []
    for item in data:
        question   = item.get("question", "").strip()
        answer     = item.get("answer", "").strip()
        hop_type   = item.get("type", "")
        sp_facts   = item.get("supporting_facts", [])   # [[title, sent_idx], ...]
        context    = {title: sentences for title, sentences in item.get("context", [])}

        if not question or not answer or len(sp_facts) < 2:
            continue

        # Collect the two gold supporting sentences
        facts = []
        for title, sent_idx in sp_facts[:2]:
            sentences = context.get(title, [])
            if sent_idx < len(sentences):
                facts.append(sentences[sent_idx].strip())

        if len(facts) < 2 or not facts[0] or not facts[1]:
            continue

        examples.append({
            "question": question,
            "answer":   answer,
            "fact1":    facts[0],
            "fact2":    facts[1],
            "type":     hop_type,
        })

        if len(examples) >= limit * 2:
            break

    return examples[:limit]


def _safe_store(content: str) -> dict:
    """Store a memory, returning {} if the content is classified as a question."""
    try:
        return add_memory(MemoryRequest(userId=USER_ID, content=content, importance=0.8))
    except Exception:
        return {}


def store_facts(fact1: str, fact2: str, pair_id: int) -> tuple[str, str]:
    """Store both supporting facts. Returns (id1, id2)."""
    r1 = _safe_store(fact1)
    time.sleep(0.05)
    r2 = _safe_store(fact2)
    return r1.get("id"), r2.get("id")


def score_result(memories: list, fact1: str, fact2: str) -> tuple[str, bool, bool]:
    """
    Returns (outcome, found_fact1, found_fact2).
    BOTH_FOUND / ONE_FOUND / NONE_FOUND.
    """
    if not memories:
        return "NONE_FOUND", False, False

    emb1 = embed(fact1)
    emb2 = embed(fact2)

    found1 = found2 = False
    for m in memories[:TOP_K]:
        mem_emb = embed(m["content"])
        if cosine(mem_emb, emb1) >= SIM_HIT:
            found1 = True
        if cosine(mem_emb, emb2) >= SIM_HIT:
            found2 = True

    if found1 and found2:
        return "BOTH_FOUND", True, True
    if found1 or found2:
        return "ONE_FOUND", found1, found2
    return "NONE_FOUND", False, False


# ── Main ──────────────────────────────────────────────────────────────────────

def run(data_path: str, limit: int, verbose: bool):
    print(f"\nHotpotQA Multi-Hop Reasoning Benchmark  |  limit={limit}  |  {data_path}")
    print("=" * 70)

    if not os.path.exists(data_path):
        print(f"\nDataset not found: {data_path}")
        print("Download with:")
        print("  wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json \\")
        print("       -O ~/hotpot_dev_distractor.json")
        sys.exit(1)

    print("Loading HotpotQA examples...", end=" ", flush=True)
    examples = load_hotpot(data_path, limit)
    print(f"{len(examples)} multi-hop examples loaded")

    results      = {"BOTH_FOUND": 0, "ONE_FOUND": 0, "NONE_FOUND": 0}
    graph_hits   = 0   # second fact found via graph (via_graph=True)
    type_counts  = {}
    t0 = time.time()

    for i, ex in enumerate(examples):
        store_facts(ex["fact1"], ex["fact2"], i)

        result   = retrieve(USER_ID, ex["question"], top_k=TOP_K)
        memories = result.get("memories", [])

        outcome, f1, f2 = score_result(memories, ex["fact1"], ex["fact2"])
        results[outcome] += 1

        hop_type = ex.get("type", "unknown")
        type_counts.setdefault(hop_type, {"BOTH_FOUND": 0, "ONE_FOUND": 0, "NONE_FOUND": 0})
        type_counts[hop_type][outcome] += 1

        # Track whether the second fact came from graph expansion
        if f2:
            for m in memories[:TOP_K]:
                if m.get("via_graph") and cosine(embed(m["content"]), embed(ex["fact2"])) >= SIM_HIT:
                    graph_hits += 1
                    break

        if verbose:
            marker = {"BOTH_FOUND": "✓✓", "ONE_FOUND": "✓·", "NONE_FOUND": "··"}[outcome]
            print(f"  [{i+1:3d}/{len(examples)}] {marker} {outcome:12s} | {ex['question'][:55]}")
        elif (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            pct = results["BOTH_FOUND"] / (i + 1) * 100
            print(f"  {i+1}/{len(examples)} done  ({elapsed:.0f}s)  "
                  f"both_found={results['BOTH_FOUND']}  ({pct:.0f}%)")

    elapsed = time.time() - t0
    total   = len(examples)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  BOTH_FOUND   (both supporting facts surfaced) : {results['BOTH_FOUND']:4d} / {total}  "
          f"({results['BOTH_FOUND']/total*100:.1f}%)")
    print(f"  ONE_FOUND    (only first fact surfaced)       : {results['ONE_FOUND']:4d} / {total}  "
          f"({results['ONE_FOUND']/total*100:.1f}%)")
    print(f"  NONE_FOUND   (neither fact surfaced)          : {results['NONE_FOUND']:4d} / {total}  "
          f"({results['NONE_FOUND']/total*100:.1f}%)")
    print(f"\n  Multi-Hop Coverage Rate                       : {results['BOTH_FOUND']/total*100:.1f}%")
    print(f"  Second fact via graph expansion               : {graph_hits} cases")

    if type_counts:
        print("\n  By question type:")
        for qtype, counts in sorted(type_counts.items()):
            n = sum(counts.values())
            pct = counts["BOTH_FOUND"] / n * 100 if n else 0
            print(f"    {qtype:20s}  both={counts['BOTH_FOUND']:3d}/{n}  ({pct:.0f}%)")

    print(f"\n  Total time: {elapsed:.1f}s")

    os.makedirs("benchmarks/results", exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "benchmark":           "HotpotQA Multi-Hop Reasoning",
        "dataset":             data_path,
        "limit":               limit,
        "timestamp":           ts,
        "results":             results,
        "multi_hop_coverage":  round(results["BOTH_FOUND"] / total, 4),
        "single_hop_partial":  round(results["ONE_FOUND"] / total, 4),
        "graph_expansion_hits": graph_hits,
        "by_type":             type_counts,
    }
    out_path = f"benchmarks/results/hotpotqa_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default=DEFAULT_DATA)
    parser.add_argument("--limit",   type=int, default=500)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(args.data, args.limit, args.verbose)
