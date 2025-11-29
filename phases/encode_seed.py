# encode_seed.py
import numpy as np

def _extract_text_from_case(case):
    """
    case می‌تواند:
      - یک رشته ساده باشد
      - یا یک dict (مثل SEED_CASES)
    این تابع سعی می‌کند متن اصلی را بیرون بکشد.
    """
    if isinstance(case, str):
        return case

    if isinstance(case, dict):
        # اولویت‌های محتمل برای کلید متن
        for key in ["text", "prompt", "content", "msg"]:
            if key in case and isinstance(case[key], str):
                return case[key]

        # اگر پیدا نشد، کل دیکشنری را رشته کنیم
        return str(case)

    # اگر نوع عجیب بود:
    return str(case)


def encode_text_to_vec(text: str):
    """
    تبدیل یک متن به سه بردار ساده A / T / I
    (نسخه‌ی خیلی ساده – فقط برای تست SEED روی C7 Core)
    """
    t = text.lower()

    audio_signal = 0
    text_signal = 0
    image_signal = 0

    # چند قاعده‌ی سمبولیک، صرفاً برای تست
    if "deep" in t or "insight" in t or "چرا" in t or "why" in t:
        text_signal += 2

    if "error" in t or "fail" in t or "بد" in t or "اشتباه" in t:
        text_signal -= 1

    if "intensity" in t or "شدت" in t or "energy" in t:
        audio_signal += 1

    if "image" in t or "picture" in t or "تصویر" in t:
        image_signal += 2

    if "gate" in t or "گیت" in t:
        text_signal += 1

    if "coherence" in t or "هماهنگی" in t:
        audio_signal += 1
        text_signal += 1

    # اگر هیچ الگوی خاصی نیافتیم، حداقل یه سیگنال متنی بده
    if audio_signal == 0 and text_signal == 0 and image_signal == 0:
        text_signal = 1

    # بردارهای ۳تایی مثل فرانت‌اندهای قبلی
    A = np.array(
        [audio_signal, audio_signal * 0.5, audio_signal * 1.2],
        dtype=float
    )
    T = np.array(
        [3 * text_signal, 3 * text_signal + 1, 3 * text_signal - 1],
        dtype=float
    )
    I = np.array(
        [image_signal - 1, image_signal, image_signal + 1],
        dtype=float
    )

    return A, T, I


def encode_case_to_emb(case):
    """
    ورودی:
      - case: می‌تواند dict یا str باشد (از SEED_CASES می‌آید)
    خروجی:
      - audio_vec, text_vec, image_vec, emb_c
    """
    raw_text = _extract_text_from_case(case)
    A, T, I = encode_text_to_vec(raw_text)
    emb = np.concatenate([A, T, I], axis=0)
    return A, T, I, emb