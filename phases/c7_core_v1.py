import numpy as np

# ============================
#  C7 Core v1.0  (One-File Demo)
#  - Embedding frontends
#  - Intensity & coherence
#  - Shallow / Deep paths
#  - Vertical memory
#  - Self-image + surprise
#  - Adaptive gate
# ============================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class C7Core:
    def __init__(self, seed: int = 42):
        rng = np.random.RandomState(seed)

        # --- Dimensions ---
        self.input_dim = 9          # 3 (audio) + 3 (text) + 3 (image)
        self.emb_dim = 16           # embedding dim after linear map
        self.shallow_hidden = 32
        self.deep_hidden = 64

        # --- Embedding layer: 9 -> 16 ---
        self.W_emb = rng.randn(self.emb_dim, self.input_dim) * 0.1
        self.b_emb = np.zeros(self.emb_dim)

        # --- A1, A3, A5 projections for coherence ---
        self.W_a1 = rng.randn(self.emb_dim) * 0.1
        self.W_a3 = rng.randn(self.emb_dim) * 0.1
        self.W_a5 = rng.randn(self.emb_dim) * 0.1
        self.b_a1 = 0.0
        self.b_a3 = 0.0
        self.b_a5 = 0.0

        # --- Shallow head ---
        # input: emb_dim + 2 (intensity, coherence)
        self.shallow_in = self.emb_dim + 2
        self.W_s1 = rng.randn(self.shallow_hidden, self.shallow_in) * 0.1
        self.b_s1 = np.zeros(self.shallow_hidden)
        self.W_s2 = rng.randn(self.shallow_hidden) * 0.1  # maps to scalar
        self.b_s2 = 0.0

        # --- Deep head ---
        # input: emb_dim + 4 (intensity, coherence, mem_a7, mem_intensity)
        self.deep_in = self.emb_dim + 4
        self.W_d1 = rng.randn(self.deep_hidden, self.deep_in) * 0.1
        self.b_d1 = np.zeros(self.deep_hidden)
        self.W_d2 = rng.randn(self.deep_hidden) * 0.1
        self.b_d2 = 0.0

        # --- Gate parameters (hand-designed, not trained here) ---
        # g = sigmoid( k1 * norm_err + k2 * user_bad - k3 * self_good_ema + k4 * intensity + kb )
        self.k1 = 2.0   # weight on normalized error
        self.k2 = 1.0   # weight on explicit user_bad
        self.k3 = 1.0   # weight on self_good_ema (confidence)
        self.k4 = 0.5   # weight on intensity
        self.kb = -1.0  # bias → پیش‌فرض بیشتر روی shallow

        # --- Self-image stats (EMA) ---
        self.base_err_ema = 1.0       # میانگین خطای شلو
        self.self_good_ema = 0.0      # چقدر خودش رو «خوب» می‌دونه
        self.ema_alpha = 0.01         # نرخ آپدیت EMA

        # --- Vertical memory (last N steps) ---
        self.mem_len = 10
        self.mem_a7 = []
        self.mem_intensity = []

    # ---------- Frontends + Collapse ----------
    def encode_frontends(self, audio, text, image):
        """
        audio, text, image: لیست / numpy array با طول 3
        خروجی: emb_c (9,) + emb (16,)
        """
        a = np.array(audio, dtype=float).reshape(-1)
        t = np.array(text, dtype=float).reshape(-1)
        i = np.array(image, dtype=float).reshape(-1)

        emb_c = np.concatenate([a, t, i])  # (9,)
        x = emb_c.reshape(self.input_dim, 1)  # (9,1)
        emb = (self.W_emb @ x).reshape(-1) + self.b_emb  # (16,)

        return emb_c, emb

    # ---------- Intensity & Coherence ----------
    def compute_intensity(self, emb_c):
        # norm-based, squashed
        norm = float(np.linalg.norm(emb_c))
        return np.tanh(norm / 10.0)

    def compute_coherence(self, emb):
        a1 = float(self.W_a1 @ emb + self.b_a1)
        a3 = float(self.W_a3 @ emb + self.b_a3)
        a5 = float(self.W_a5 @ emb + self.b_a5)
        vals = np.array([a1, a3, a5], dtype=float)
        var = float(np.var(vals))
        coherence = 1.0 / (1.0 + var)
        return coherence, a1, a3, a5

    # ---------- Shallow / Deep paths ----------
    def shallow_path(self, emb, intensity, coherence):
        x = np.concatenate([emb, np.array([intensity, coherence])])
        h = np.tanh(self.W_s1 @ x + self.b_s1)
        y = float(self.W_s2 @ h + self.b_s2)
        return y

    def deep_path(self, emb, intensity, coherence):
        # حافظه عمودی
        if len(self.mem_a7) > 0:
            mem_a7 = float(np.mean(self.mem_a7))
            mem_int = float(np.mean(self.mem_intensity))
        else:
            mem_a7 = 0.0
            mem_int = intensity

        x = np.concatenate([emb,
                            np.array([intensity, coherence, mem_a7, mem_int])])
        h = np.tanh(self.W_d1 @ x + self.b_d1)
        y = float(self.W_d2 @ h + self.b_d2)
        return y

    # ---------- Gate ----------
    def compute_gate(self, base_err, intensity, user_bad):
        # EMA روی خطای شلو
        self.base_err_ema = (1 - self.ema_alpha) * self.base_err_ema + self.ema_alpha * base_err

        # EMA روی self_good → وقتی خطا کوچک است، self_good بالا می‌رود
        # self_good ≈ exp(-|error|)
        local_good = np.exp(-abs(base_err))
        self.self_good_ema = (1 - self.ema_alpha) * self.self_good_ema + self.ema_alpha * local_good

        # نرمال‌سازی خطا
        norm_err = base_err / (self.base_err_ema + 1e-6)

        # سیگنال user_bad: اگر None بود، 0 در نظر می‌گیریم
        user_bad_val = 0.0 if user_bad is None else float(user_bad)

        # gate logit
        logit = (self.k1 * norm_err +
                 self.k2 * user_bad_val -
                 self.k3 * self.self_good_ema +
                 self.k4 * intensity +
                 self.kb)

        g = float(sigmoid(logit))
        return g, norm_err

    # ---------- Memory Update ----------
    def update_memory(self, a7_final, intensity):
        self.mem_a7.append(float(a7_final))
        self.mem_intensity.append(float(intensity))
        if len(self.mem_a7) > self.mem_len:
            self.mem_a7.pop(0)
        if len(self.mem_intensity) > self.mem_len:
            self.mem_intensity.pop(0)

    # ---------- Forward ----------
    def forward(self, audio, text, image, target=None, user_bad=None):
        """
        یک پاس کامل مغز:
        - فرانت‌اندها
        - intensity & coherence
        - shallow / deep
        - gate
        - memory update
        """
        emb_c, emb = self.encode_frontends(audio, text, image)
        intensity = self.compute_intensity(emb_c)
        coherence, a1, a3, a5 = self.compute_coherence(emb)

        # اگر target داده نشده بود → target = sum(Emb-C)
        if target is None:
            target_val = float(np.sum(emb_c))
        else:
            target_val = float(target)

        y_shallow = self.shallow_path(emb, intensity, coherence)
        base_err = abs(y_shallow - target_val)

        y_deep = self.deep_path(emb, intensity, coherence)

        g, norm_err = self.compute_gate(base_err, intensity, user_bad)

        # ترکیب نهایی
        y_final = (1.0 - g) * y_shallow + g * y_deep
        final_err = y_final - target_val

        # آپدیت حافظه
        self.update_memory(y_final, intensity)

        return {
            "emb_c": emb_c,
            "emb": emb,
            "intensity": intensity,
            "coherence": coherence,
            "A1": a1,
            "A3": a3,
            "A5": a5,
            "y_shallow": y_shallow,
            "y_deep": y_deep,
            "gate": g,
            "y_final": y_final,
            "target": target_val,
            "base_err": base_err,
            "final_err": final_err,
            "norm_err": norm_err,
            "self_good_ema": self.self_good_ema,
            "base_err_ema": self.base_err_ema,
            "mem_a7_len": len(self.mem_a7),
        }


# ============================
#  Demo Run (fixed sample)
# ============================

if __name__ == "__main__":
    core = C7Core(seed=123)

    # همان ورودی کلاسیک تست‌هامان:
    audio = [2, 1, 2]
    text  = [3, 3, 3]
    image = [-1, 0, 1]

    # target = sum(Emb-C) بعد از collapse
    out = core.forward(audio, text, image, target=None, user_bad=None)

    print("=== C7 Core v1.0 – Demo Run ===")
    print(f"Emb-C        : {out['emb_c']}")
    print(f"Intensity    : {out['intensity']:.3f}")
    print(f"Coherence    : {out['coherence']:.3f}")
    print(f"A1,A3,A5     : {out['A1']:.3f}, {out['A3']:.3f}, {out['A5']:.3f}")
    print("-----")
    print(f"y_shallow    : {out['y_shallow']:.3f}")
    print(f"y_deep       : {out['y_deep']:.3f}")
    print(f"gate g       : {out['gate']:.3f}")
    print(f"y_final      : {out['y_final']:.3f}")
    print(f"target       : {out['target']:.3f}")
    print(f"base_err     : {out['base_err']:.3f}")
    print(f"final_err    : {out['final_err']:.3f}")
    print("-----")
    print(f"norm_err     : {out['norm_err']:.3f}")
    print(f"self_good_ema: {out['self_good_ema']:.3f}")
    print(f"base_err_ema : {out['base_err_ema']:.3f}")
    print(f"mem_a7_len   : {out['mem_a7_len']}")