"""Pure verse ID retrieval test.

For each question that has a known verse ID:
  A) Query = original question alone
  B) Query = "BhG X.Y: original question"

Then check: did the correct verse chunk appear in top-K results?
Only verse-retrieval metrics — no semantic comparison.
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import json, random, re, time
from pathlib import Path


CHAPTER_VERSES = [46, 72, 43, 42, 29, 47, 30, 28, 34, 42, 55, 20, 35, 27, 20, 24, 28, 78]

_SUPPLEMENTED_VERSES = {
    (1,38), (1,47), (2,35), (5,9), (10,26), (12,6), (12,7),
    (15,6), (15,9), (15,13), (15,16), (16,2), (16,3), (16,9),
    (16,13), (17,4), (17,14),
}

def unique_key_to_verse_ref(key: int) -> str:
    cum = 0
    for ch, count in enumerate(CHAPTER_VERSES, 1):
        if key <= cum + count:
            return f"BhG {ch}.{key - cum}"
        cum += count
    return ""

def _parse_ref(ref: str) -> tuple:
    parts = ref.replace("BhG ", "").split(".")
    if len(parts) == 2:
        try:
            return (int(parts[0]), int(parts[1]))
        except ValueError:
            pass
    return (0, 0)

def _is_valid_ref(ref: str) -> bool:
    ch, v = _parse_ref(ref)
    if ch < 1 or ch > 18 or v < 1:
        return False
    if v <= CHAPTER_VERSES[ch - 1]:
        return True
    return (ch, v) in _SUPPLEMENTED_VERSES

def _filter_valid(refs: list[str]) -> list[str]:
    return [r for r in refs if _is_valid_ref(r)]

def extract_verse_refs_from_text(text: str) -> set[str]:
    refs = set()
    for m in re.finditer(r'(?:BhG|BG)\s+(\d+\.\d+)', text, re.IGNORECASE):
        refs.add(f"BhG {m.group(1)}")
    for m in re.finditer(r'Chapter\s+(\d+)[,\s]+Verse\s+(\d+)', text, re.IGNORECASE):
        refs.add(f"BhG {m.group(1)}.{m.group(2)}")
    return refs


def load_gita_guidance_qa(path: str) -> list[dict]:
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            msgs = d.get('messages', [])
            if len(msgs) >= 2:
                q = msgs[0].get('content', '')
                a = msgs[1].get('content', '')
                if q and a:
                    refs = _filter_valid(sorted(extract_verse_refs_from_text(a)))
                    if refs:
                        pairs.append({'question': q, 'source': 'gita_guidance_qa', 'verse_refs': refs})
    return pairs

def load_hf_gita_qa(path: str) -> list[dict]:
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            q, a, ch, vs = d.get('question',''), d.get('answer',''), d.get('chapter_no'), d.get('verse_no')
            if q and a and ch and vs:
                ref = f"BhG {ch}.{vs}"
                if _is_valid_ref(ref):
                    pairs.append({'question': q, 'source': 'hf_gita_qa', 'verse_refs': [ref]})
    return pairs

def load_kaggle_gita_qa(path: str) -> list[dict]:
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            q, a, vs = d.get('question',''), d.get('answer',''), d.get('verse_source','')
            if q and a and vs and '.' in vs:
                ref = f"BhG {vs}"
                if _is_valid_ref(ref):
                    pairs.append({'question': q, 'source': 'kaggle_gita_qa', 'verse_refs': [ref]})
    return pairs

def load_iskcon(path: str, n: int) -> list[dict]:
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            key, trans = d.get('unique_key', 0), d.get('translation', '')
            if key and trans:
                ref = unique_key_to_verse_ref(key)
                if ref:
                    pairs.append({'question': "Explain this Bhagavad Gita verse.",
                                  'source': 'iskcon_vedabase', 'verse_refs': [ref]})
    random.Random(42).shuffle(pairs)
    return pairs[:n]


def check_verse_in_reranked(expected_refs: list[str], reranked: list[dict]) -> dict:
    expected = set(expected_refs)
    verse_results = [r for r in reranked if r.get("chunk_type") == "verse"]
    out = {"expected": sorted(expected), "retrieved_top10": [r.get("verse_ref","") for r in verse_results[:10]]}
    for k in [1, 3, 5, 10]:
        top = {r.get("verse_ref","") for r in verse_results[:k]}
        out[f"recall_at_{k}"] = len(top & expected) / max(len(expected), 1)
    out["mrr"] = 0.0
    for i, r in enumerate(verse_results):
        if r.get("verse_ref","") in expected:
            out["mrr"] = 1.0 / (i + 1)
            break
    return out


def run(samples: int, iskcon_n: int):
    print("=" * 70)
    print("PURE VERSE ID RETRIEVAL TEST (with Neo4j Graph)")
    print("Compares: question alone vs question + 'BhG X.Y:' prefix")
    print("Metric:  did the correct verse chunk appear in top-K?")
    print(f"Samples: {samples} per dataset, ISKCON: {iskcon_n}")
    print("=" * 70)

    from src.utils.config import Config
    from src.langchain_components.graph import SRAGGraphPipeline

    config = Config()
    pipeline = SRAGGraphPipeline(config)
    pipeline.preprocess()
    pipeline.build_indices()

    # ── Neo4j graph ──
    graph_available = False
    try:
        pipeline._get_graph_retriever()
        graph_available = True
        print("  Neo4j: ONLINE")
    except Exception as e:
        print(f"  Neo4j: OFFLINE ({e})")

    print(f"Chunks: {len(pipeline.chunks)}")
    if pipeline.vector_store.index:
        print(f"FAISS: {pipeline.vector_store.index.ntotal}")
    print(f"BM25: {len(pipeline.bm25_retriever.chunk_ids)}")

    # Datasets
    ed = Path("data/evaluation/external")
    ds = {}
    for name, fn, loader, extra in [
        ("gita_guidance_qa", "gita_guidance_qa.jsonl", load_gita_guidance_qa, None),
        ("hf_gita_qa", "hf_gita_qa.jsonl", load_hf_gita_qa, None),
        ("kaggle_gita_qa", "kaggle_gita_qa.jsonl", load_kaggle_gita_qa, None),
        ("iskcon_vedabase", "iskcon_vedabase.jsonl", load_iskcon, iskcon_n),
    ]:
        p = ed / fn
        if p.exists():
            data = loader(str(p), extra) if extra is not None else loader(str(p))
            if data:
                ds[name] = data
                print(f"{name}: {len(data)} pairs")

    all_results = {}
    # Stage tracking: (reranker, fusion, direct)
    total_no = {"s1":0,"s3":0,"s5":0,"s10":0,"mrr":0,"n":0}
    total_wi = {"s1":0,"s3":0,"s5":0,"s10":0,"mrr":0,"n":0, "direct":0}
    total_fi = {"s1":0,"s3":0,"s5":0,"s10":0,"mrr":0,"n":0}  # fusion-only with ID

    for ds_name, pairs in ds.items():
        sample = random.Random(42).sample(pairs, min(samples, len(pairs)))
        print(f"\n── {ds_name} ({len(sample)}) ──")

        ds_out = []
        for i, qa in enumerate(sample):
            q = qa['question']
            refs = qa['verse_refs']
            vid = refs[0] if refs else ""

            conds = [("without_id", q)]
            if vid:
                conds.append(("with_id", f"{vid}: {q}"))

            row = {"question": q[:80], "verse_refs": refs}
            for cl, qt in conds:
                try:
                    t0 = time.time()
                    result = pipeline.query(qt, use_api=False, retrieval_only=True)

                    rr = result.get("reranked_results", [])
                    fused = result.get("fused_results", [])
                    intermediate = result.get("intermediate", {})
                    verse_ref_detected = intermediate.get("verse_ref_detected", False)

                    vm = check_verse_in_reranked(refs, rr)
                    fm = check_verse_in_reranked(refs, fused) if fused else {}
                    row[cl] = {
                        "time": round(time.time()-t0, 2),
                        "verse_retrieval": {k: v for k, v in vm.items() if k != "expected"},
                        "fusion_only": {k: v for k, v in fm.items() if k != "expected"},
                        "verse_ref_direct": verse_ref_detected,
                    }
                except Exception as e:
                    row[cl] = {"error": str(e)}

            ds_out.append(row)

            nv = row.get("without_id",{}).get("verse_retrieval",{})
            wv = row.get("with_id",{}).get("verse_retrieval",{})
            nr1 = nv.get("recall_at_1", -1)
            wr1 = wv.get("recall_at_1", -1)
            has_dir = "✓" if row.get("with_id",{}).get("verse_ref_direct") else "✗"
            print(f"  [{i+1}] {vid:<12} R@1: {nr1:.2f}/{wr1:.2f} dir:{has_dir}")

        all_results[ds_name] = ds_out
        valid = [r for r in ds_out if "without_id" in r and "error" not in r["without_id"]]
        valid_w = [r for r in ds_out if "with_id" in r and "error" not in r["with_id"]]

        def avg_fn(lst, cond, path):
            vals = []
            for r in lst:
                d = r.get(cond, {})
                for k in path:
                    d = d.get(k, {}) if isinstance(d, dict) else {}
                if isinstance(d, (int, float)):
                    vals.append(d)
            return sum(vals)/len(vals) if vals else 0.0

        print(f"\n  ── {ds_name} Summary (reranker/fusion/direct) ──")
        print(f"  {'Metric':<15} {'NoID-rerank':>12} {'WithID-rerank':>14} {'WithID-fuse':>12} {'Dir%':>6}")
        print(f"  {'-'*15} {'-'*12} {'-'*14} {'-'*12} {'-'*6}")
        for m in ["recall_at_1","recall_at_3","recall_at_5","recall_at_10","mrr"]:
            nv = avg_fn(valid, "without_id", ["verse_retrieval", m])
            wv = avg_fn(valid_w, "with_id", ["verse_retrieval", m])
            fv = avg_fn(valid_w, "with_id", ["fusion_only", m])
            dir_pct = sum(1 for r in valid_w if r["with_id"].get("verse_ref_direct")) / max(len(valid_w), 1) * 100
            print(f"  {m:<15} {nv:>12.4f} {wv:>14.4f} {fv:>12.4f} {dir_pct:>5.0f}%")

        for r in valid:
            v = r["without_id"]["verse_retrieval"]
            total_no["s1"] += v["recall_at_1"]; total_no["s3"] += v["recall_at_3"]
            total_no["s5"] += v["recall_at_5"]; total_no["s10"] += v["recall_at_10"]
            total_no["mrr"] += v["mrr"]; total_no["n"] += 1
        for r in valid_w:
            v = r["with_id"]["verse_retrieval"]
            total_wi["s1"] += v["recall_at_1"]; total_wi["s3"] += v["recall_at_3"]
            total_wi["s5"] += v["recall_at_5"]; total_wi["s10"] += v["recall_at_10"]
            total_wi["mrr"] += v["mrr"]; total_wi["n"] += 1
            if r["with_id"].get("verse_ref_direct"):
                total_wi["direct"] += 1
            f = r["with_id"].get("fusion_only", {})
            if f:
                total_fi["s1"] += f.get("recall_at_1", 0); total_fi["s3"] += f.get("recall_at_3", 0)
                total_fi["s5"] += f.get("recall_at_5", 0); total_fi["s10"] += f.get("recall_at_10", 0)
                total_fi["mrr"] += f.get("mrr", 0); total_fi["n"] += 1

    # Overall
    n = max(total_no["n"], 1)
    wn = max(total_wi["n"], 1)
    fn = max(total_fi["n"], 1)
    print(f"\n{'='*70}")
    print("OVERALL — VERSE ID RETRIEVAL ACCURACY (3 stages)")
    print(f"{'='*70}")
    print(f"  {'Metric':<15} {'NoID-rerank':>12} {'WithID-rerank':>14} {'WithID-fuse':>12} {'Dir%':>6}")
    print(f"  {'-'*15} {'-'*12} {'-'*14} {'-'*12} {'-'*6}")
    for m, k in [("Recall@1","s1"),("Recall@3","s3"),("Recall@5","s5"),("Recall@10","s10"),("MRR","mrr")]:
        nv = total_no[k] / n
        wv = total_wi[k] / wn
        fv = total_fi[k] / fn
        dp = total_wi["direct"] / wn * 100
        print(f"  {m:<15} {nv:>12.4f} {wv:>14.4f} {fv:>12.4f} {dp:>5.0f}%")

    print(f"\n  Neo4j: {'ONLINE' if graph_available else 'OFFLINE'}")
    print(f"  N={n} (no ID), {wn} (with ID)")
    print(f"  Dir% = % of verse_ref lookups found in Neo4j graph")

    # Save
    out = {
        "config": {"samples_per_dataset": samples, "iskcon_samples": iskcon_n, "neo4j_online": graph_available},
        "overall": {
            "without_verse_id": {k: round(total_no[k]/n, 4) for k in ["s1","s3","s5","s10","mrr"]},
            "with_verse_id_reranker": {k: round(total_wi[k]/wn, 4) for k in ["s1","s3","s5","s10","mrr"]},
            "with_verse_id_fusion_only": {k: round(total_fi[k]/fn, 4) for k in ["s1","s3","s5","s10","mrr"]},
            "verse_ref_direct_pct": round(total_wi["direct"]/wn*100, 1),
        },
        "by_dataset": {},
        "detailed_results": all_results,
    }
    for dn, dr in all_results.items():
        v = [r for r in dr if "without_id" in r and "error" not in r["without_id"]]
        w = [r for r in dr if "with_id" in r and "error" not in r["with_id"]]
        def avg_one(lst, cond, m, sub="verse_retrieval"):
            vals = []
            for r in lst:
                d = r.get(cond,{}).get(sub,{})
                if isinstance(d.get(m), (int,float)):
                    vals.append(d[m])
            return round(sum(vals)/len(vals), 4) if vals else 0
        d_pct = sum(1 for r in w if r["with_id"].get("verse_ref_direct")) / max(len(w),1)*100 if w else 0
        out["by_dataset"][dn] = {
            "without_verse_id": {m: avg_one(v, "without_id", m) for m in ["recall_at_1","recall_at_3","recall_at_5","recall_at_10","mrr"]},
            "with_verse_id_reranker": {m: avg_one(w, "with_id", m) for m in ["recall_at_1","recall_at_3","recall_at_5","recall_at_10","mrr"]},
            "with_verse_id_fusion_only": {m: avg_one(w, "with_id", m, "fusion_only") for m in ["recall_at_1","recall_at_3","recall_at_5","recall_at_10","mrr"]},
            "verse_ref_direct_pct": round(d_pct, 1),
        }

    p = Path("data/evaluation/verse_lookup_test.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {p}")

    if graph_available:
        pipeline.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--iskcon-samples", type=int, default=15)
    run(p.parse_args().samples, p.parse_args().iskcon_samples)
