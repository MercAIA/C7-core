import numpy as np
import math

class GroundedBrain:
    """
    C7 – Phase 9 (Grounded)
    -----------------------
    - Frontend ثابت برای سه مدالیته (Audio / Text / Image)
    - سه آرایه A1, A3, A5 → خروجی shallow
    - یک head عمیق (W_deep, b_deep)
    - Survival Core = مبنای بلندمدت |خطا|  (EWMA)
    - گیت بین shallow / deep بر اساس تهدید نسبت به Survival Core
    """

    def __init__(self, lr=5e-4, sc_beta=0.01, k_threat=3.0, seed=0):
        rng = np.random.default_rng(seed)

        # آرایه‌های افقی
        self.W1 = rng.normal(0, 0.1, size=9)
        self.b1 = 0.0

        self.W3 = rng.normal(0, 0.1, size=9)
        self.b3 = 0.0

        self.W5 = rng.normal(0, 0.1, size=9)
        self.b5 = 0.0

        # هد عمیق
        self.W_deep = rng.normal(0, 0.1, size=9)
        self.b_deep = 0.0

        # هایپرپارامترها
        self.lr = lr
        self.sc_beta = sc_beta        # سرعت آپدیت Survival Core
        self.k_threat = k_threat      # حساسیت گیت به تهدید

        # مبدا بقا: مبنای بلندمدت |خطا|
        self.survival_core = 1.0

        # تاریخچه‌ها (فقط برای دیباگ / لاگ)
        self.history_abs_err = []
        self.history_intensity = []
        self.history_coherence = []

    # ---------- FRONTEND & STATS ----------

    def frontend(self, a, t, i):
        """
        مپ ساده ورودی‌ها به فضای frontend
        (اینجا عمداً trainable نیست، فقط نقش scale داره)
        """
        a_f = np.array(a, dtype=float)          # audio
        t_f = np.array(t, dtype=float) * 3.0    # text را قوی‌تر می‌کنیم
        i_f = np.array(i, dtype=float)          # image
        return a_f, t_f, i_f

    def emb_and_intensity(self, a, t, i):
        """
        Emb-C و شدت (intensity) را بر اساس انرژی مدالیته‌ها می‌سازد.
        intensity = max_energy / (sum_energies)
        """
        a_f, t_f, i_f = self.frontend(a, t, i)
        emb = np.concatenate([a_f, t_f, i_f])

        ea = np.linalg.norm(a_f)
        et = np.linalg.norm(t_f)
        ei = np.linalg.norm(i_f)

        total = ea + et + ei + 1e-8
        intensity = max(ea, et, ei) / total
        return emb, intensity

    # ---------- FORWARD ----------

    def forward(self, a, t, i, update_survival=True, target=None):
        """
        یک پاس کامل:
        - emb, intensity
        - A1, A3, A5 → y_shallow
        - y_deep
        - گیت بر اساس تهدید نسبت به Survival Core
        - خروجی نهایی y
        """

        emb, intensity = self.emb_and_intensity(a, t, i)

        # سه آرایه افقی
        A1 = float(emb @ self.W1 + self.b1)
        A3 = float(emb @ self.W3 + self.b3)
        A5 = float(emb @ self.W5 + self.b5)
        y_shallow = (A1 + A3 + A5) / 3.0

        vals = np.array([A1, A3, A5], dtype=float)
        var = float(np.var(vals))
        coherence = 1.0 / (1.0 + var)  # هرچه آرایه‌ها هم‌نظرتر → کوهرنس بالاتر

        # تارگت: جمع Emb-C
        if target is None:
            target = float(np.sum(emb))

        # خطای shallow
        base_err = y_shallow - target
        abs_base = abs(base_err)

        # ========= SURVIVAL CORE =========
        # تهدید = چقدر وضعیت فعلی از مبنای بقا بدتر است
        threat = abs_base - self.survival_core
        g = 1.0 / (1.0 + math.exp(-self.k_threat * threat))  # گیت بین 0 و 1

        # هد عمیق
        y_deep = float(emb @ self.W_deep + self.b_deep)

        # ترکیب shallow / deep
        y = (1.0 - g) * y_shallow + g * y_deep
        err = y - target

        # آپدیت Survival Core (مبدا بقا)
        if update_survival:
            self.survival_core = (1.0 - self.sc_beta) * self.survival_core + \
                                 self.sc_beta * abs(err)
            self.history_abs_err.append(abs(err))
            self.history_intensity.append(intensity)
            self.history_coherence.append(coherence)

        return {
            "emb": emb,
            "A1": A1,
            "A3": A3,
            "A5": A5,
            "y_shallow": y_shallow,
            "y_deep": y_deep,
            "y": y,
            "target": target,
            "err": err,
            "base_err": base_err,
            "g": g,
            "intensity": intensity,
            "coherence": coherence,
        }

    # ---------- TRAIN STEP ----------

    def train_step(self, a, t, i):
        """
        گرادیان ساده فقط روی:
          W1, b1, W3, b3, W5, b5, W_deep, b_deep
        Frontend ثابت است.
        """

        out = self.forward(a, t, i, update_survival=True)
        emb = out["emb"]
        err = out["err"]
        g = out["g"]

        # dL/dy
        dL_dy = err  # چون loss = 0.5 * err^2

        # y = (1-g)*y_shallow + g*y_deep  (g را ثابت فرض می‌کنیم)
        dy_dys = (1.0 - g)
        dy_dyd = g

        dL_dys = dL_dy * dy_dys
        dL_dyd = dL_dy * dy_dyd

        # y_shallow = (A1 + A3 + A5) / 3
        dL_dA1 = dL_dys / 3.0
        dL_dA3 = dL_dys / 3.0
        dL_dA5 = dL_dys / 3.0

        # A_k = emb · W_k + b_k
        # ∂L/∂W_k = dL/dA_k * emb
        # ∂L/∂b_k = dL/dA_k
        self.W1 -= self.lr * dL_dA1 * emb
        self.b1 -= self.lr * dL_dA1

        self.W3 -= self.lr * dL_dA3 * emb
        self.b3 -= self.lr * dL_dA3

        self.W5 -= self.lr * dL_dA5 * emb
        self.b5 -= self.lr * dL_dA5

        # هد عمیق
        self.W_deep -= self.lr * dL_dyd * emb
        self.b_deep -= self.lr * dL_dyd

        return out

# ---------- HELPER برای دیتا تصادفی ----------

def random_vec(low=-1, high=4, size=3):
    return [np.random.randint(low, high) for _ in range(size)]

# ---------- MAIN ----------

if __name__ == "__main__":
    np.random.seed(0)

    brain = GroundedBrain(lr=5e-4, sc_beta=0.01, k_threat=3.0, seed=42)

    steps = 500
    print("Starting GroundedBrain Phase 9 training for", steps, "steps...\n")

    for step in range(1, steps + 1):
        a = random_vec()
        t = random_vec()
        i = random_vec()

        out = brain.train_step(a, t, i)

        if step % 50 == 0:
            print(
                f"Step {step:4d} | "
                f"err:{out['err']:+6.3f}  "
                f"|g:{out['g']:.3f}  "
                f"int:{out['intensity']:.3f}  "
                f"coh:{out['coherence']:.3f}  "
                f"SC:{brain.survival_core:.3f}"
            )

    print("\nTraining finished.\n")

    # ---------- EVAL روی یک ورودی ثابت ----------

    test_a = [2, 1, 2]
    test_t = [3, 3, 3]
    test_i = [-1, 0, 1]

    eval_out = brain.forward(test_a, test_t, test_i, update_survival=False)

    emb = eval_out["emb"]
    target = eval_out["target"]
    y_shallow = eval_out["y_shallow"]
    y_deep = eval_out["y_deep"]
    y = eval_out["y"]
    g = eval_out["g"]

    print("========== EVAL ON FIXED SAMPLE ==========")
    print("Audio Input :", test_a)
    print("Text  Input :", test_t)
    print("Image Input :", test_i)
    print()
    print("Emb-C Collapsed Vector:", emb.tolist())
    print(f"Target(sum Emb-C): {target:8.3f}")
    print()
    print(f"Intensity       : {eval_out['intensity']:.3f}")
    print(f"Coherence       : {eval_out['coherence']:.3f}")
    print(f"Survival Core   : {brain.survival_core:.3f}")
    print()
    print(f"y_shallow       : {y_shallow:8.3f}")
    print(f"y_deep          : {y_deep:8.3f}")
    print(f"gate g          : {g:8.3f}")
    print(f"A7_final (y)    : {y:8.3f}")
    print(f"Final error     : {eval_out['err']:8.3f}")
    print("==========================================\n")

    # ---------- RANDOM EVAL (10 نمونه) ----------

    print("========== RANDOM EVAL (10 samples) ==========")
    abs_errs = []
    for idx in range(1, 11):
        a = random_vec()
        t = random_vec()
        i = random_vec()
        out = brain.forward(a, t, i, update_survival=False)
        abs_err = abs(out["err"])
        abs_errs.append(abs_err)

        mode = "DEEP" if out["g"] > 0.5 else "SHALLOW"
        print(
            f"#{idx:2d} | int:{out['intensity']:.3f} "
            f"coh:{out['coherence']:.3f} g:{out['g']:.3f} "
            f"mode:{mode:7s} target:{out['target']:6.2f} "
            f"A7:{out['y']:6.2f} abs_err:{abs_err:5.3f}"
        )

    print("----------------------------------------------")
    print(f"Mean |error| over 10 samples: {np.mean(abs_errs):.3f}")
    print(f"Max  |error| over 10 samples: {np.max(abs_errs):.3f}")
    print("==============================================")