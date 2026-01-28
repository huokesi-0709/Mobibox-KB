from __future__ import annotations
import argparse

from monibox_kb.config import settings
from monibox_kb.runtime.rag_engine import RagEngine
from monibox_kb.runtime.protocol_engine import ProtocolEngine
from monibox_kb.runtime.hardware_iface import MockHardware
from monibox_kb.runtime.safety_guard import SafetyGuard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--auto_top_tags", type=int, default=2)
    ap.add_argument("--events", default="", help="模拟事件，逗号分隔，如 imu_strong_shake")
    args = ap.parse_args()

    events = [e.strip() for e in args.events.split(",") if e.strip()]

    hw = MockHardware()
    rag = RagEngine(settings.rag_db_path)
    prot = ProtocolEngine()
    guard = SafetyGuard()

    # 1) 路由得到 tags（用于协议触发，也用于RAG过滤）
    rr = rag.router.route(args.q, top_tags=args.auto_top_tags)

    # 2) 协议优先
    hit = prot.match(args.q, rr.tags, events)
    if hit:
        print("\n[PROTOCOL HIT]", hit["protocol_id"], hit["name"])
        for a in hit.get("actions", []):
            t = a.get("type")

            if t == "tts":
                raw_text = a.get("text", "")
                res = guard.check(raw_text)

                if res.level == "allow":
                    hw.tts(res.safe_text, style=a.get("style"))
                elif res.level == "rewrite":
                    print("[GUARD rewrite]", res.reasons)
                    hw.tts(res.safe_text, style=a.get("style"))
                else:
                    # block
                    print("[GUARD block]", res.reasons)
                    hw.tts(res.safe_text, style="urgent_calm")

            elif t == "led":
                hw.led(a.get("pattern", {}))
            elif t == "screen":
                hw.screen(a.get("text", ""), ms=int(a.get("ms", 2000)))
        return

    # 3) 未命中协议：RAG 兜底
    print("\n[NO PROTOCOL] fallback to RAG")
    res_list = rag.auto_search(args.q, topk=args.topk, auto_top_tags=args.auto_top_tags)

    for i, r in enumerate(res_list, start=1):
        print(f"\n[{i}] {r.display_id} ({r.dimension}/{r.risk})")
        print(f"    dist={r.distance:.6f} final={r.final_distance:.6f}")

        # 每条输出也走护栏
        res = guard.check(r.text)
        if res.level == "allow":
            hw.tts(res.safe_text, style="calm_clear")
        elif res.level == "rewrite":
            print("[GUARD rewrite]", res.reasons)
            hw.tts(res.safe_text, style="calm_clear")
        else:
            print("[GUARD block]", res.reasons)
            hw.tts(res.safe_text, style="urgent_calm")


if __name__ == "__main__":
    main()