import numpy as np
import os
import json

# ===============================
#  Phase 8 - C7 + ObserverPrime
# ===============================

# ------- Helper functions -------

def l2_energy(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.sum(x * x)) + 1e-9)

def coherence_from_arrays(a1, a3, a5):
    vals = np.array([a1, a3, a5], dtype=float)
    var = float(np.var(vals))
    return 1.0 / (1.0 + var), var

# ------- Frontends (Audio / Text / Image) -------

class AudioFrontend:
    """
    ساده: هر مؤلفه +1 تا با الگوهای قبلی سازگار باشه:
    [2,1,2] -> [3,2,3]
    """
    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        return x + 1.0

class TextFrontend:
    """
    الگو نزدیک به لاگ‌های قبلی:
    text_out = 3*x + 3
    [3,3,3] -> [12,12,12]
    """
    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        return 3.0 * x + 3.0

class ImageFrontend:
    """
    image_out = 2*(x+1)
    [-1,0,1] -> [0,2,4]
    """
    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        return 2.0 * (x + 1.0)

# ------- Prism-7 Core (A1, A3, A5, A7) -------

class Prism7:
    def __init__(self, dim=9, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.dim = dim
        # وزن‌های سه آرایه اصلی
        self.w1 = rng.normal(0, 0.1, size=dim)
        self.w3 = rng.normal(0, 0.1, size=dim)
        self.w5 = rng.normal(0, 0.1, size=dim)

    def forward(self, emb_c):
        """
        emb_c: وکتور 9تایی Emb-C
        """
        emb_c = np.asarray(emb_c, dtype=float)
        a1 = float(np.dot(self.w1, emb_c))
        a3 = float(np.dot(self.w3, emb_c))
        a5 = float(np.dot(self.w5, emb_c))
        a7_raw = (a1 + a3 + a5) / 3.0
        return a1, a3, a5, a7_raw

    def backward(self, emb_c, target, a1, a3, a5, a7_raw, lr=1e-3):
        """
        گرادیان ساده برای نزدیک شدن a7_raw به target:
        loss = 0.5 * (a7_raw - target)^2
        dL/dWi = (a7_raw - target) * (1/3) * emb_c
        """
        emb_c = np.asarray(emb_c, dtype=float)
        err = a7_raw - target
        grad = (err / 3.0) * emb_c
        self.w1 -= lr * grad
        self.w3 -= lr * grad
        self.w5 -= lr * grad
        return float(err)

# ------- ObserverPrime (فاز ۸) -------

class ObserverPrime:
    """
    ناظر سطح بالا:
      - تاریخچه کوتاه از (intensity, error, coherence)
      - تعیین mode A/B/C
      - تعیین gain، mem_depth، smooth
      - سیکل چهارتایی phase_quadrant = 0..3
    """

    def __init__(self, history_len=20):
        self.history_len = history_len
        self.history = []  # list of dicts
        self.cycle_step = 0
        self.last_a7 = None

    def update_history(self, intensity, error, coherence):
        self.history.append({
            "intensity": float(intensity),
            "error": float(error),
            "coherence": float(coherence),
        })
        if len(self.history) > self.history_len:
            self.history.pop(0)

    def stats(self):
        if not self.history:
            return 0.0, 0.0, 0.0
        intens = np.array([h["intensity"] for h in self.history], dtype=float)
        errs = np.array([h["error"] for h in self.history], dtype=float)
        cohs = np.array([h["coherence"] for h in self.history], dtype=float)
        return float(np.mean(intens)), float(np.mean(errs)), float(np.mean(cohs))

    def decide_mode(self, intensity, error, coherence):
        """
        قوانین ساده ولی الهام‌گرفته از چیزایی که گفتی:
          - error کوچک + intensity پایین -> mode A (آرام / کم‌عمق)
          - error متوسط + intensity متوسط -> mode B (متعادل)
          - error بزرگ یا intensity بالا -> mode C (حفاظتی / عمیق)
        """
        abs_err = abs(error)

        if abs_err < 0.5 and intensity < 0.4:
            mode = "A"
            base_gain = 0.95
            mem_depth = 3
            smooth = 0.08
        elif abs_err < 1.5 and intensity < 0.75:
            mode = "B"
            base_gain = 1.00
            mem_depth = 5
            smooth = 0.12
        else:
            mode = "C"
            base_gain = 1.05
            mem_depth = 8
            smooth = 0.22

        # سیکل چهارتایی (مثل فصل‌ها / فاز ماه)
        self.cycle_step += 1
        phase_quadrant = self.cycle_step % 4

        # modulation کوچک بر اساس فاز:
        #   0: کمی کاهش gain (استراحت)
        #   1: حالت نرمال
        #   2: کمی افزایش gain (فاز فعال)
        #   3: کمی کاهش دوباره برای برگشت
        if phase_quadrant == 0:
            gain = base_gain * 0.97
        elif phase_quadrant == 2:
            gain = base_gain * 1.03
        else:
            gain = base_gain

        # depth_permission شبیه همون چیزی که قبلا داشتیم
        # اینجا coherence هم دخیل می‌کنیم: coherence بالاتر => عمق بیشتر منطقی‌تره
        depth_permission = float(
            np.clip(
                0.3 * intensity + 0.4 * (abs_err / (1.0 + abs_err)) + 0.3 * coherence,
                0.0,
                1.0,
            )
        )

        return {
            "mode": mode,
            "gain": float(gain),
            "mem_depth": int(mem_depth),
            "smooth": float(smooth),
            "phase_quadrant": int(phase_quadrant),
            "depth_permission": depth_permission,
        }

    def regulate_output(self, a7_raw, a1, a3, a5, config):
        """
        self-reg افقی + عمودی:
          - اعمال gain روی A1/A3/A5
          - گرفتن میانگین
          - اسموت کردن با last_a7 (مثل ناخودآگاه تنظیم‌کننده)
        """
        gain = config["gain"]
        smooth = config["smooth"]

        a1_scaled = a1 * gain
        a3_scaled = a3 * gain
        a5_scaled = a5 * gain
        a7_new = (a1_scaled + a3_scaled + a5_scaled) / 3.0

        if self.last_a7 is None:
            a7_final = a7_new
        else:
            a7_final = (1.0 - smooth) * self.last_a7 + smooth * a7_new

        self.last_a7 = a7_final
        return float(a7_final), float(a1_scaled), float(a3_scaled), float(a5_scaled)

# ------- C7 Brain V8 -------

class C7BrainV8:
    def __init__(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        self.audio_f = AudioFrontend()
        self.text_f = TextFrontend()
        self.image_f = ImageFrontend()
        self.prism = Prism7(dim=9, rng=rng)
        self.observer = ObserverPrime()

        # برای گزارش نهایی
        self.memory_a7 = []
        self.memory_intensity = []

    def forward(self, audio_in, text_in, image_in):
        # فرانت‌اندها
        A = self.audio_f(audio_in)
        T = self.text_f(text_in)
        I = self.image_f(image_in)

        # Emb-C
        emb_c = np.concatenate([A, T, I], axis=0)

        # شدت و انرژی‌ها
        e_audio = l2_energy(A)
        e_text = l2_energy(T)
        e_image = l2_energy(I)
        total_e = e_audio + e_text + e_image
        intensity = e_text / (total_e + 1e-9)  # غالباً text غالب بود

        # Prism-7
        a1, a3, a5, a7_raw = self.prism.forward(emb_c)

        # coherence افقی
        coherence, var_arrays = coherence_from_arrays(a1, a3, a5)

        return {
            "A": A,
            "T": T,
            "I": I,
            "emb_c": emb_c,
            "intensity": intensity,
            "energies": {
                "audio": e_audio,
                "text": e_text,
                "image": e_image,
            },
            "a1": a1,
            "a3": a3,
            "a5": a5,
            "a7_raw": a7_raw,
            "coherence": coherence,
            "var_arrays": var_arrays,
        }

    def train_step(self, lr=1e-3):
        # ورودی‌های تصادفی شبیه فازهای قبل
        audio_in = self.rng.integers(-1, 3, size=3)
        text_in = self.rng.integers(-1, 4, size=3)
        image_in = self.rng.integers(-1, 3, size=3)

        f = self.forward(audio_in, text_in, image_in)
        emb_c = f["emb_c"]
        intensity = f["intensity"]
        coherence = f["coherence"]

        # هدف: جمع Emb-C (مثل فازهای قبل)
        target = float(np.sum(emb_c))

        # خطا قبل از self-reg
        err_raw = self.prism.backward(
            emb_c,
            target,
            f["a1"],
            f["a3"],
            f["a5"],
            f["a7_raw"],
            lr=lr,
        )

        # آپدیت history برای Observer
        self.observer.update_history(intensity=intensity,
                                     error=err_raw,
                                     coherence=coherence)
        hist_int, hist_err, hist_coh = self.observer.stats()
        config = self.observer.decide_mode(intensity, err_raw, coherence)

        # self-reg روی خروجی
        a7_final, a1_scaled, a3_scaled, a5_scaled = self.observer.regulate_output(
            f["a7_raw"], f["a1"], f["a3"], f["a5"], config
        )

        # ذخیره در حافظه زمانی
        self.memory_a7.append(a7_final)
        self.memory_intensity.append(intensity)
        if len(self.memory_a7) > 50:
            self.memory_a7.pop(0)
            self.memory_intensity.pop(0)

        loss = 0.5 * (f["a7_raw"] - target) ** 2
        return {
            "loss": float(loss),
            "err_raw": float(err_raw),
            "config": config,
            "hist_stats": {
                "avg_intensity": hist_int,
                "avg_error": hist_err,
                "avg_coherence": hist_coh,
            },
        }

    def test_on_reference(self):
        """
        همون ورودی رفرنس قدیمی:
          audio = [2,1,2]
          text  = [3,3,3]
          image = [-1,0,1]
        """
        audio_in = np.array([2, 1, 2])
        text_in = np.array([3, 3, 3])
        image_in = np.array([-1, 0, 1])

        f = self.forward(audio_in, text_in, image_in)
        emb_c = f["emb_c"]
        target = float(np.sum(emb_c))

        # تصمیم Observer روی این تک ورودی
        # برای این که error داشته باشیم، a7_raw - target:
        err_now = f["a7_raw"] - target
        config = self.observer.decide_mode(
            intensity=f["intensity"],
            error=err_now,
            coherence=f["coherence"],
        )
        a7_final, a1_scaled, a3_scaled, a5_scaled = self.observer.regulate_output(
            f["a7_raw"], f["a1"], f["a3"], f["a5"], config
        )

        return {
            "audio_in": audio_in.tolist(),
            "text_in": text_in.tolist(),
            "image_in": image_in.tolist(),
            "A": f["A"].tolist(),
            "T": f["T"].tolist(),
            "I": f["I"].tolist(),
            "emb_c": emb_c.tolist(),
            "intensity": float(f["intensity"]),
            "energies": f["energies"],
            "a1": float(f["a1"]),
            "a3": float(f["a3"]),
            "a5": float(f["a5"]),
            "a7_raw": float(f["a7_raw"]),
            "coherence": float(f["coherence"]),
            "target": target,
            "err_raw": float(err_now),
            "config": config,
            "a1_scaled": a1_scaled,
            "a3_scaled": a3_scaled,
            "a5_scaled": a5_scaled,
            "a7_final": a7_final,
            "final_error": float(a7_final - target),
        }

# ------- Training script (فاز ۸) -------

def main():
    rng = np.random.default_rng()
    brain = C7BrainV8(rng=rng)

    STEPS = 300
    print("Starting Phase 8 training for", STEPS, "steps...\n")

    for step in range(1, STEPS + 1):
        r = brain.train_step(lr=8e-4)
        if step % 50 == 0:
            cfg = r["config"]
            print(
                f"Step {step:3d}/{STEPS} - "
                f"loss:{r['loss']:.4f} "
                f"err_raw:{r['err_raw']:.3f} "
                f"mode:{cfg['mode']} "
                f"coh_avg:{r['hist_stats']['avg_coherence']:.3f} "
                f"int_avg:{r['hist_stats']['avg_intensity']:.3f}"
            )

    print("\nPhase 8 training finished.\n")

    # تست روی ورودی رفرنس
    result = brain.test_on_reference()

    print("========== TEST AFTER PHASE 8 ==========\n")
    print("=== Raw Inputs ===")
    print("Audio Input :", result["audio_in"])
    print("Text Input  :", result["text_in"])
    print("Image Input :", result["image_in"])
    print("\n=== Frontend Outputs ===")
    print("AudioFrontend Output :", result["A"])
    print("TextFrontend Output  :", result["T"])
    print("ImageFrontend Output :", result["I"])
    print("\n=== Emb-C & Intensity ===")
    print("Emb-C Collapsed Vector:", result["emb_c"])
    print(f"Intensity             : {result['intensity']:.3f}")
    print("Energies              :", result["energies"])
    print("\n=== Arrays & A7 (RAW) ===")
    print(f"A1       : {result['a1']:.4f}")
    print(f"A3       : {result['a3']:.4f}")
    print(f"A5       : {result['a5']:.4f}")
    print(f"A7_raw   : {result['a7_raw']:.4f}")
    print(f"Target   : {result['target']:.4f}")
    print(f"Err_raw  : {result['err_raw']:.4f}")
    print(f"Coherence: {result['coherence']:.3f}")
    print("\n=== ObserverPrime Config ===")
    cfg = result["config"]
    print("Mode            :", cfg["mode"])
    print("Phase quadrant  :", cfg["phase_quadrant"])
    print(f"Gain            : {cfg['gain']:.3f}")
    print("Mem depth       :", cfg["mem_depth"])
    print(f"Smooth          : {cfg['smooth']:.3f}")
    print(f"Depth permission: {cfg['depth_permission']:.3f}")
    print("\n=== Arrays After Self-Regulation ===")
    print(f"A1_scaled: {result['a1_scaled']:.4f}")
    print(f"A3_scaled: {result['a3_scaled']:.4f}")
    print(f"A5_scaled: {result['a5_scaled']:.4f}")
    print(f"A7_final : {result['a7_final']:.4f}")
    print(f"Final error (A7_final - target): {result['final_error']:.4f}")
    print("\n=========================================")


if __name__ == "__main__":
    main()