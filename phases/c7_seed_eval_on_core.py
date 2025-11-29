# c7_seed_eval_on_core.py

import numpy as np
from c7_core_v1 import C7Core       # کلاس مغز اصلی
from encode_seed import encode_case_to_emb
from c7_seed_chat_cases import SEED_CASES


def _case_text_preview(case):
    """خلاصه‌ای از متن کیس برای چاپ در لاگ."""
    if isinstance(case, dict):
        for key in ["text", "prompt", "content", "msg"]:
            if key in case and isinstance(case[key], str):
                return case[key]
        return str(case)
    return str(case)


def _run_core_forward(brain, audio, text, image):
    """
    خروجی forward رو به‌صورت انعطاف‌پذیر هندل می‌کنیم:
    - اگر dict بود: سعی می‌کنیم y_shallow, y_deep, y_final رو از کلیدها بخونیم
    - اگر tuple/list بود: فرض می‌گیریم سه تای اول y_s, y_d, y_f هستن
    - اگر چیز دیگه بود: فقط y_final رو ازش می‌گیریم
    """
    out = brain.forward(audio, text, image)

    # حالت ۱: دیکشنری (خیلی وقت‌ها اینطوری طراحی می‌کنیم)
    if isinstance(out, dict):
        y_shallow = float(out.get("y_shallow", 0.0))
        y_deep    = float(out.get("y_deep", 0.0))
        # اگر y_final نباشه، y_deep رو به‌عنوان خروجی نهایی می‌گیریم
        y_final   = float(out.get("y_final", y_deep))
        return y_shallow, y_deep, y_final

    # حالت ۲: لیست/تاپل (مثلاً ۱۸ تا مقدار مختلف)
    if isinstance(out, (tuple, list)):
        if len(out) >= 3:
            y_shallow = float(out[0])
            y_deep    = float(out[1])
            y_final   = float(out[2])
        else:
            # اگر کمتر از ۳ تا بود، هرچی هست رو حداقل برای y_final استفاده می‌کنیم
            y_shallow = 0.0
            y_deep    = 0.0
            y_final   = float(out[0]) if len(out) > 0 else 0.0
        return y_shallow, y_deep, y_final

    # حالت ۳: هر چیز دیگه (مثلاً فقط یک عدد)
    y_final = float(out)
    return 0.0, 0.0, y_final


def eval_seed_cases():
    brain = C7Core()

    print("=== C7 Core v1 – Seed Chat EVAL ===")
    abs_errors = []

    for i, case in enumerate(SEED_CASES, start=1):
        # چهار خروجی: سه بردار جدا + emb_c
        audio, text, image, emb_c = encode_case_to_emb(case)

        # ✅ این‌جا دیگه forward رو امن و انعطاف‌پذیر صدا می‌زنیم
        y_shallow, y_deep, y_final = _run_core_forward(brain, audio, text, image)

        # Target:
        if isinstance(case, dict) and "target" in case:
            target = float(case["target"])
        else:
            target = float(np.sum(emb_c))

        err = y_final - target
        abs_err = abs(err)
        abs_errors.append(abs_err)

        preview = _case_text_preview(case)
        preview = preview.replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:77] + "..."

        print(f"Case {i:02d}:")
        print(f"  text    : {preview!r}")
        print(f"  target  : {target:.3f}")
        print(f"  y_final : {y_final:.3f}   (y_s={y_shallow:.3f}, y_d={y_deep:.3f})")
        print(f"  err     : {err:.3f}   |err|={abs_err:.3f}")
        print("-" * 60)

    if abs_errors:
        print("=== SUMMARY ===")
        print(f"Mean |err| over {len(abs_errors)} cases: {np.mean(abs_errors):.3f}")
        print(f"Max  |err|: {np.max(abs_errors):.3f}")


if __name__ == "__main__":
    eval_seed_cases()