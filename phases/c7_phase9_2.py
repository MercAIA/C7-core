import numpy as np

class C7BrainV9_2:
    """
    Phase 9.2 – Observer distilled into the core head.
    No explicit observer layer at inference time.
    """

    def __init__(self, lr=0.001, alpha=0.5, seed=42):
        np.random.seed(seed)
        self.lr = lr
        self.alpha = alpha   # وزنِ ترمِ «ناظرِ درونی»
        # هد اصلی: A7_raw = w_out · emb + b_out
        self.w_out = np.random.randn(9) * 0.1
        self.b_out = 0.0

    # ---------- Frontends (مثل نسخه‌های قبلی، ساده و قابل ردیابی) ----------

    def audio_frontend(self, a):
        v = np.array(a, dtype=float)
        # کمی mix ساده
        return np.array([
            v[0] + 0.5 * v[1],
            v[1] + 0.5 * v[2],
            v[2] + 0.5 * v[0],
        ])

    def text_frontend(self, t):
        v = np.array(t, dtype=float)
        # متن را قوی‌تر می‌کنیم
        base = np.array([
            v[0] + v[1],
            v[1] + v[2],
            v[0] + v[2],
        ])
        return base * 2.0

    def image_frontend(self, i):
        v = np.array(i, dtype=float)
        # کمی نرم‌تر
        return np.array([
            0.5 * (v[0] + v[1]),
            0.5 * (v[1] + v[2]),
            0.5 * (v[0] + v[2]),
        ])

    # ---------- Emb-C, Prism-7 و هد ----------

    def embed_c(self, a_out, t_out, i_out):
        return np.concatenate([a_out, t_out, i_out])

    def prism7_arrays(self, emb):
        # سه آرایه عمودی: هر کدوم جمع یک بلوک از Emb-C
        a1 = float(np.sum(emb[0:3]))
        a3 = float(np.sum(emb[3:6]))
        a5 = float(np.sum(emb[6:9]))
        return a1, a3, a5

    def intensity(self, emb):
        norm = np.linalg.norm(emb)
        return norm / (norm + 1.0)

    def forward(self, a, t, i):
        a_out = self.audio_frontend(a)
        t_out = self.text_frontend(t)
        i_out = self.image_frontend(i)
        emb = self.embed_c(a_out, t_out, i_out)

        a1, a3, a5 = self.prism7_arrays(emb)
        a7_raw = float(np.dot(self.w_out, emb) + self.b_out)

        info = {
            "audio_out": a_out,
            "text_out": t_out,
            "image_out": i_out,
            "emb": emb,
            "a1": a1,
            "a3": a3,
            "a5": a5,
            "a7_raw": a7_raw,
            "intensity": self.intensity(emb),
        }
        return a7_raw, info

    # ---------- آموزش با distillation ناظر ----------

    def train_step(self, a, t, i):
        # forward
        a7_raw, info = self.forward(a, t, i)
        emb = info["emb"]
        a1, a3, a5 = info["a1"], info["a3"], info["a5"]

        target = float(np.sum(emb))
        teacher = (a1 + a3 + a5) / 3.0   # اثر ناظرِ قدیمی داخل خودِ معماری

        err_target = a7_raw - target
        err_teacher = a7_raw - teacher

        loss = err_target**2 + self.alpha * (err_teacher**2)

        # گرادیان ساده نسبت به w_out و b_out
        dL_dA7 = 2.0 * err_target + 2.0 * self.alpha * err_teacher
        dA7_dw = emb
        dA7_db = 1.0

        self.w_out -= self.lr * dL_dA7 * dA7_dw
        self.b_out -= self.lr * dL_dA7 * dA7_db

        # coherence فقط برای مانیتور (نه در گرادیان)
        var_arrays = np.var([a1, a3, a5])
        coherence = 1.0 / (1.0 + var_arrays)

        return {
            "loss": float(loss),
            "err_target": float(err_target),
            "err_teacher": float(err_teacher),
            "coherence": float(coherence),
            "intensity": float(info["intensity"]),
        }

    def train(self, steps=300):
        for step in range(1, steps + 1):
            # ورودی تصادفی در بازه [-1, 3]
            a = np.random.randint(-1, 4, size=3)
            t = np.random.randint(-1, 4, size=3)
            i = np.random.randint(-1, 4, size=3)

            stats = self.train_step(a, t, i)

            if step == 1 or step % 50 == 0:
                print(
                    f"Step {step:3d}/{steps} - "
                    f"loss:{stats['loss']:.4f} "
                    f"err_t:{stats['err_target']:.3f} "
                    f"err_teacher:{stats['err_teacher']:.3f} "
                    f"coh:{stats['coherence']:.3f} "
                    f"int:{stats['intensity']:.3f}"
                )

    # ---------- تست روی ورودی ثابت ----------

    def test_fixed(self):
        a = [2, 1, 2]
        t = [3, 3, 3]
        i = [-1, 0, 1]

        a7_raw, info = self.forward(a, t, i)
        emb = info["emb"]
        a1, a3, a5 = info["a1"], info["a3"], info["a5"]
        intensity = info["intensity"]

        target = float(np.sum(emb))
        teacher = (a1 + a3 + a5) / 3.0
        err_target = a7_raw - target
        err_teacher = a7_raw - teacher
        var_arrays = np.var([a1, a3, a5])
        coherence = 1.0 / (1.0 + var_arrays)

        print("\n========== TEST AFTER PHASE 9.2 (DISTILLED) ==========\n")
        print("=== Raw Inputs ===")
        print("Audio Input :", a)
        print("Text  Input :", t)
        print("Image Input :", i)
        print("\n=== Frontend Outputs ===")
        print("AudioFrontend Output :", info['audio_out'])
        print("TextFrontend  Output :", info['text_out'])
        print("ImageFrontend Output :", info['image_out'])

        print("\n=== Emb-C & Intensity ===")
        print("Emb-C Collapsed Vector:", emb)
        print(f"Target(sum Emb-C):      {target:.4f}")
        print(f"Teacher(mean A1,A3,A5): {teacher:.4f}")
        print(f"Intensity:              {intensity:.3f}")

        print("\n=== Arrays & A7 ===")
        print(f"A1       : {a1:.3f}")
        print(f"A3       : {a3:.3f}")
        print(f"A5       : {a5:.3f}")
        print(f"A7_raw   : {a7_raw:.3f}")
        print(f"Err_target (A7 - target):  {err_target:.3f}")
        print(f"Err_teacher(A7 - meanA):   {err_teacher:.3f}")
        print(f"Coherence (1/(1+var)):     {coherence:.3f}")

        print("\n(no external observer layer; A7_final = A7_raw)")
        print("=====================================================\n")


if __name__ == "__main__":
    brain = C7BrainV9_2(lr=0.001, alpha=0.5, seed=42)
    print("Starting Phase 9.2 (distilled observer) training...\n")
    brain.train(steps=300)
    brain.test_fixed()