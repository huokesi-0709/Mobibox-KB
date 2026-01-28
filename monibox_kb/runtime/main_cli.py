import argparse
from monibox_kb.config import settings
from monibox_kb.runtime.rag_engine import RagEngine

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--auto_top_tags", type=int, default=2)
    args = ap.parse_args()

    eng = RagEngine(settings.rag_db_path)
    res = eng.auto_search(args.q, topk=args.topk, auto_top_tags=args.auto_top_tags)

    for i, r in enumerate(res, start=1):
        print(f"\n[{i}] {r.display_id}  ({r.dimension}/{r.risk})")
        print(f"    dist={r.distance:.6f} final={r.final_distance:.6f} score={r.quality_score} status={r.status}")
        print(f"    text={r.text}")

if __name__ == "__main__":
    main()