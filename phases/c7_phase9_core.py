import numpy as np
import pickle
import os

# ================================
#   Core Class — Phase 9 (SELF-EVOLUTION, STABLE)
# ================================

class C7BrainV9Core:
    def __init__(self):
        # سه آرایه عصبی (A1, A3, A5)
        self.weights = np.random.randn(3, 9) * 0.1

        # حافظه‌ی خطا برای الگوی شکست/موفقیت
        self.error_history = []

        # سلف‌ ایمیج که با تجربه خودش تغییر می‌کند
        self.self_image = {
            "stability":  0.5,   # حس پایداری
            "confidence": 0.5,   # حس توانایی
            "depth_bias": 0.5    # تمایل به پردازش عمیق
        }

        # شدت داخلی (وابسته به تجربه خطا)
        self.intensity = 0.5

    # -------------------------------
    #   Frontend ها
    # -------------------------------
    def frontend_audio(self, x):
        return np.array([x[0] + 1, x[1] * 1.5, x[2] + 1], dtype=float)

    def frontend_text(self, x):
        return np.array([x[0] * 4, x[1] * 4, x[2] * 4], dtype=float)

    def frontend_image(self, x):
        return np.array([x[0] * 0.5, x[1] * 1.0, x[2] * 1.0], dtype=float)

    # -------------------------------
    #   Intensity حسی
    # -------------------------------
    def compute_intensity(self, emb):
        e_audio = np.linalg.norm(emb[:3])
        e_text  = np.linalg.norm(emb[3:6])
        e_img   = np.linalg.norm(emb[6:])
        e = e_audio + e_text + e_img
        return float(min(1.0, e / 50.0))

    # -------------------------------
    #   Forward pass
    # -------------------------------
    def forward(self, emb):
        A = self.weights.dot(emb)        # [A1, A3, A5]
        A7 = float(A.mean())
        coherence = float(1.0 / (1.0 + np.var(A)))
        return A, A7, coherence

    # -------------------------------
    #   Local trigger (ناراضی بودن موضعی)
    # -------------------------------
    def trigger_local(self, error):
        """
        اگر جواب بد بود → شدت داخلی ↑ (پردازش عمیق‌تر برای بعدی‌ها)
        اگر خوب بود → شدت کمی ↓
        """
        if abs(error) > 1.0:
            self.intensity += 0.05
        else:
            self.intensity -= 0.02

        self.intensity = float(np.clip(self.intensity, 0.0, 1.0))

    # -------------------------------
    #   Evolution trigger (الگوی شکست)
    # -------------------------------
    def trigger_evolution(self):
        """
        چند خطای پشت‌سرهم بزرگ → self_image تغییر می‌کند.
        """
        if len(self.error_history) < 8:
            return

        last_mean = float(np.mean(np.abs(self.error_history[-8:])))

        if last_mean > 2.0:
            # شکست‌های تکرار شونده
            self.self_image["confidence"] *= 0.95
            self.self_image["depth_bias"]  *= 1.10
            self.self_image["stability"]   *= 0.97
        else:
            # روند نسبتا خوب
            self.self_image["confidence"] *= 1.05
            self.self_image["stability"]  *= 1.03

        # کلیپ هویت
        for k in self.self_image:
            self.self_image[k] = float(np.clip(self.self_image[k], 0.1, 3.0))

    # -------------------------------
    #   Update weights (پایدار شده)
    # -------------------------------
    def update(self, emb, target, A7):
        error = A7 - target

        # گرادیان را کلیپ و نرمال می‌کنیم که وزنه‌ها منفجر نشوند
        err_clipped = float(np.clip(error, -20.0, 20.0))

        # نرمال‌سازی emb برای کنترل scale
        emb_norm = float(np.linalg.norm(emb) + 1e-6)
        emb_unit = emb / emb_norm

        # نرخ یادگیری کوچک و ثابت (وابسته به self_image/intensity نیست)
        base_lr = 0.001
        lr = base_lr

        grad = err_clipped * emb_unit  # جهت اصلاح

        # همان گرادیان برای هر سه آرایه (A1,A3,A5)
        grad_stack = np.vstack([grad, grad, grad])
        self.weights -= lr * grad_stack

        return float(error)

    # -------------------------------
    #   Training step
    # -------------------------------
    def train_step(self, A_in, T_in, I_in):
        # 1) frontends
        A = self.frontend_audio(A_in)
        T = self.frontend_text(T_in)
        I = self.frontend_image(I_in)

        emb = np.concatenate([A, T, I]).astype(float)

        sensory_intensity = self.compute_intensity(emb)

        # 2) forward
        A_arr, A7, coh = self.forward(emb)
        target = float(emb.sum())

        # 3) update weights
        error = self.update(emb, target, A7)

        # 4) local trigger (بر اساس همین جواب)
        self.trigger_local(error)

        # 5) evolution memory
        self.error_history.append(error)
        if len(self.error_history) > 200:
            self.error_history.pop(0)

        # 6) evolution trigger (الگوی چند قدم اخیر)
        self.trigger_evolution()

        return {
            "err": float(error),
            "A7": float(A7),
            "int_sensory": sensory_intensity,
            "int_internal": float(self.intensity),
            "coh": float(coh),
            "selfimg": self.self_image.copy()
        }


# ================================
#   TRAINING LOOP
# ================================

if __name__ == "__main__":
    brain = C7BrainV9Core()

    for step in range(1, 301):
        A = np.random.randint(-1, 3, 3)
        T = np.random.randint(-1, 3, 3)
        I = np.random.randint(-1, 3, 3)

        out = brain.train_step(A, T, I)

        if step % 50 == 0:
            print(
                f"Step {step:3d} | "
                f"err:{out['err']:.3f}  "
                f"coh:{out['coh']:.3f}  "
                f"int_s:{out['int_sensory']:.3f}  "
                f"int_i:{out['int_internal']:.3f}  "
                f"conf:{out['selfimg']['confidence']:.2f}  "
                f"depth:{out['selfimg']['depth_bias']:.2f}"
            )

    # ذخیره‌ی مغز
    with open("c7_phase9_core_brain.pkl", "wb") as f:
        pickle.dump(brain, f)

    print("\nTraining finished.")