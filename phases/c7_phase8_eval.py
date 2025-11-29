import numpy as np

# =========================
#  C7 – Phase 8 Eval Model
# =========================

# ساده: هر سه فرانت‌اند فعلاً همون ورودی رو به float برمی‌گردونن
# (با آموزش، لایه‌های آخر خودشون رو با این فضا هماهنگ می‌کنن)


class AudioFrontend:
    def __call__(self, x):
        return np.array(x, dtype=float)


class TextFrontend:
    def __call__(self, x):
        return np.array(x, dtype=float)


class ImageFrontend:
    def __call__(self, x):
        return np.array(x, dtype=float)


def compute_energies(a_vec, t_vec, i_vec):
    """محاسبه انرژی هر مدالیته + شدت (intensity) بر اساس انرژی غالب."""
    e_audio = float(np.linalg.norm(a_vec) + 1e-8)
    e_text = float(np.linalg.norm(t_vec) + 1e-8)
    e_image = float(np.linalg.norm(i_vec) + 1e-8)

    total = e_audio + e_text + e_image
    energies = {"audio": e_audio, "text": e_text, "image": e_image}

    dominant = max(energies, key=energies.get)
    intensity = energies[dominant] / total if total > 0 else 0.0

    return energies, intensity, dominant


def coherence_from_arrays(a1, a3, a5):
    """Coherence ساده: وارونگیِ واریانس سه آرایه."""
    arr = np.array([a1, a3, a5], dtype=float)
    var = float(np.var(arr))
    coherence = 1.0 / (1.0 + var)
    return coherence, var


class C7BrainPhase8:
    """
    مدل خطی ساده:
      - Emb-C با ابعاد 9 -> سه آرایه A1, A3, A5
      - A7_raw = میانگین سه آرایه
      - self-reg زمان‌محور با memory کوتاه
    """

    def __init__(self, lr=1e-3, mem_depth=10, alpha=0.1):
        self.lr = lr
        self.mem_depth = mem_depth
        self.alpha = alpha

        # وزن‌های سه آرایه (9 ورودی -> 1 خروجی)
        self.w1 = np.random.randn(9) * 0.1
        self.b1 = 0.0

        self.w3 = np.random.randn(9) * 0.1
        self.b3 = 0.0

        self.w5 = np.random.randn(9) * 0.1
        self.b5 = 0.0

        # حافظه A7 برای self-reg
        self.memory = []

    def forward_arrays(self, emb):
        """خروجی خام آرایه‌ها + A7_raw و coherence."""
        a1 = float(np.dot(self.w1, emb) + self.b1)
        a3 = float(np.dot(self.w3, emb) + self.b3)
        a5 = float(np.dot(self.w5, emb) + self.b5)

        a7_raw = (a1 + a3 + a5) / 3.0
        coherence, var = coherence_from_arrays(a1, a3, a5)
        return a1, a3, a5, a7_raw, coherence, var

    def self_regulate(self, a7_raw):
        """linear temporal smoothing روی A7_raw با استفاده از حافظه کوتاه."""
        if len(self.memory) == 0:
            a7_final = a7_raw
        else:
            mem_avg = float(np.mean(self.memory))
            a7_final = (1.0 - self.alpha) * a7_raw + self.alpha * mem_avg

        # حافظه را آپدیت کن
        self.memory.append(a7_final)
        if len(self.memory) > self.mem_depth:
            self.memory.pop(0)

        return a7_final

    def train_step(self, emb, target):
        """یک گام آموزش ساده (MSE روی A7_raw)."""
        # forward
        a1 = float(np.dot(self.w1, emb) + self.b1)
        a3 = float(np.dot(self.w3, emb) + self.b3)
        a5 = float(np.dot(self.w5, emb) + self.b5)
        a7_raw = (a1 + a3 + a5) / 3.0

        error = a7_raw - target
        loss = error ** 2

        # گرادیان‌ها (مدل کاملاً خطی است)
        dL_da7 = 2.0 * error
        da7_da1 = 1.0 / 3.0
        da7_da3 = 1.0 / 3.0
        da7_da5 = 1.0 / 3.0

        dL_da1 = dL_da7 * da7_da1
        dL_da3 = dL_da7 * da7_da3
        dL_da5 = dL_da7 * da7_da5

        # d( w·x + b ) / dw = x  , d(...)/db = 1
        grad_w1 = dL_da1 * emb
        grad_b1 = dL_da1

        grad_w3 = dL_da3 * emb
        grad_b3 = dL_da3

        grad_w5 = dL_da5 * emb
        grad_b5 = dL_da5

        # آپدیت وزن‌ها
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1

        self.w3 -= self.lr * grad_w3
        self.b3 -= self.lr * grad_b3

        self.w5 -= self.lr * grad_w5
        self.b5 -= self.lr * grad_b5

        return float(loss), float(error)

    def reset_memory(self):
        self.memory = []


def random_input_triplet():
    """
    سه ورودی تصادفی برای audio / text / image.
    مقادیر بین -1 و 3 (شبیه فازهای قبلی).
    """
    audio = np.random.randint(-1, 4, size=3).tolist()
    text = np.random.randint(-1, 4, size=3).tolist()
    image = np.random.randint(-1, 4, size=3).tolist()
    return audio, text, image


def build_frontends():
    return AudioFrontend(), TextFrontend(), ImageFrontend()


def run_single_sample(brain, af, tf, imf, audio, text, image):
    """یک پاس کامل از ورودی تا A7_final + متریک‌ها."""
    # frontends
    a_vec = af(audio)
    t_vec = tf(text)
    i_vec = imf(image)

    # Emb-C
    emb = np.concatenate([a_vec, t_vec, i_vec])
    target = float(np.sum(emb))

    energies, intensity, dom = compute_energies(a_vec, t_vec, i_vec)
    a1, a3, a5, a7_raw, coherence, var = brain.forward_arrays(emb)
    a7_final = brain.self_regulate(a7_raw)

    err_final = a7_final - target

    return {
        "audio": audio,
        "text": text,
        "image": image,
        "emb": emb,
        "target": target,
        "energies": energies,
        "intensity": float(intensity),
        "dominant": dom,
        "a1": a1,
        "a3": a3,
        "a5": a5,
        "a7_raw": a7_raw,
        "a7_final": a7_final,
        "coherence": float(coherence),
        "var": float(var),
        "err_final": float(err_final),
    }


def main():
    np.random.seed(42)

    # 1) ساخت فرانت‌اندها و مغز
    af, tf, imf = build_frontends()
    brain = C7BrainPhase8(lr=1e-3, mem_depth=10, alpha=0.1)

    # 2) آموزش روی n_steps نمونه تصادفی
    n_steps = 300
    print("Starting Phase 8 EVAL training for", n_steps, "steps...\n")
    losses = []

    for step in range(1, n_steps + 1):
        audio, text, image = random_input_triplet()
        a_vec = af(audio)
        t_vec = tf(text)
        i_vec = imf(image)
        emb = np.concatenate([a_vec, t_vec, i_vec])
        target = float(np.sum(emb))

        loss, err = brain.train_step(emb, target)
        losses.append(loss)

        if step % 50 == 0:
            avg_loss = float(np.mean(losses[-50:]))
            print(
                f"Step {step:3d}/{n_steps} - avg loss:{avg_loss:.4f} last_err:{err:.3f}"
            )

    print("\nTraining finished.\n")

    # 3) تست روی نمونه ثابت (همون نمونه کلاسیک [2,1,2], [3,3,3], [-1,0,1])
    brain.reset_memory()
    print("========== TEST ON FIXED SAMPLE ==========\n")

    fixed_audio = [2, 1, 2]
    fixed_text = [3, 3, 3]
    fixed_image = [-1, 0, 1]

    res_fixed = run_single_sample(
        brain, af, tf, imf, fixed_audio, fixed_text, fixed_image
    )

    print("Audio Input :", res_fixed["audio"])
    print("Text Input  :", res_fixed["text"])
    print("Image Input :", res_fixed["image"])
    print("\nEmb-C Collapsed Vector:", res_fixed["emb"])
    print("Target(sum Emb-C):     ", f"{res_fixed['target']:.4f}")
    print("Intensity              :", f"{res_fixed['intensity']:.3f}")
    print("Energies               :", res_fixed["energies"])
    print("Dominant modality      :", res_fixed["dominant"])
    print("\nA1, A3, A5:", f"{res_fixed['a1']:.3f}",
          f"{res_fixed['a3']:.3f}", f"{res_fixed['a5']:.3f}")
    print("A7_raw     :", f"{res_fixed['a7_raw']:.3f}")
    print("A7_final   :", f"{res_fixed['a7_final']:.3f}")
    print("Coherence  :", f"{res_fixed['coherence']:.3f}")
    print("Final error:", f"{res_fixed['err_final']:.3f}")
    print("\n=========================================\n")

    # 4) تست روی چند ورودی مختلف (Multi-input eval)
    brain.reset_memory()
    n_eval = 10  # اگر خواستی می‌تونی این رو بزاری 50 یا 100
    print(f"========== MULTI-INPUT EVAL ({n_eval} samples) ==========\n")

    all_errors = []
    all_coh = []
    all_int = []

    for i in range(1, n_eval + 1):
        audio, text, image = random_input_triplet()

        res = run_single_sample(brain, af, tf, imf, audio, text, image)
        all_errors.append(abs(res["err_final"]))
        all_coh.append(res["coherence"])
        all_int.append(res["intensity"])

        print(f"Step {i}:")
        print("  inputs    : A=", res["audio"],
              "T=", res["text"], "I=", res["image"])
        print("  intensity : ", f"{res['intensity']:.3f}",
              " dom:", res["dominant"])
        print("  A7_raw    : ", f"{res['a7_raw']:.3f}")
        print("  A7_final  : ", f"{res['a7_final']:.3f}")
        print("  target    : ", f"{res['target']:.3f}")
        print("  err_final : ", f"{res['err_final']:.3f}")
        print("  coherence : ", f"{res['coherence']:.3f}")
        print("  var(A1,A3,A5):", f"{res['var']:.3f}")
        print("-" * 72)

    mean_abs_err = float(np.mean(all_errors))
    max_abs_err = float(np.max(all_errors))
    mean_coh = float(np.mean(all_coh))
    mean_int = float(np.mean(all_int))

    print("\n========== SUMMARY OVER", n_eval, "SAMPLES ==========")
    print("Mean |error_final| :", f"{mean_abs_err:.4f}")
    print("Max  |error_final| :", f"{max_abs_err:.4f}")
    print("Mean coherence     :", f"{mean_coh:.4f}")
    print("Mean intensity     :", f"{mean_int:.4f}")
    print("===============================================")


if __name__ == "__main__":
    main()