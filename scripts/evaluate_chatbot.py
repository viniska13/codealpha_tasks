"""Measure the FAQ chatbot on tuning + held-out queries.

    python scripts/evaluate_chatbot.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.faq_engine import FAQEngine, evaluate  # noqa: E402

if __name__ == "__main__":
    engine = FAQEngine()
    # eval_queries.json was used while tuning thresholds; holdout_queries.json was not.
    for name in ("eval_queries", "holdout_queries"):
        r = evaluate(engine, json.loads((ROOT / "data" / f"{name}.json").read_text(encoding="utf-8")))
        print(f"\n== {name} ==")
        print(f"Top-1 accuracy             : {r['top1']:.1%}")
        print(f"Top-3 accuracy             : {r['top3']:.1%}")
        print(f"Confident & correct answer : {r['confident_correct']:.1%}")
        print(f"Out-of-scope not answered  : {r['out_of_scope_rejected']:.1%}")
        for m in r["misses"]:
            print(f"  MISS  {m['query']!r} expected={m['expected']} got={m['got']} ({m['score']})")
        for l in r["leaks"]:
            print(f"  LEAK  {l['query']!r} answered with {l['answered_with']} ({l['score']})")
