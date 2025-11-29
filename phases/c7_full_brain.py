import numpy as np


# ----- Shallow core: simple linear mapping over Emb-C -----

class ShallowCore:
    def __init__(self, dim_in: int):
        rng = np.random.default_rng(42)
        self.w = rng.normal(0.0, 0.1, size=(dim_in,))
        self.b = 0.0

    def __call__(self, emb_c: np.ndarray) -> float:
        x = emb_c.astype(float)
        y = float(self.w @ x + self.b)
        return y


# ----- Deep core: small MLP over Emb-C -----

class DeepCore:
    def __init__(self, dim_in: int, dim_hidden: int = 16):
        rng = np.random.default_rng(123)
        self.W1 = rng.normal(0.0, 0.1, size=(dim_hidden, dim_in))
        self.b1 = np.zeros((dim_hidden,))
        self.W2 = rng.normal(0.0, 0.1, size=(1, dim_hidden))
        self.b2 = np.zeros((1,))

    def __call__(self, emb_c: np.ndarray) -> float:
        x = emb_c.astype(float)
        h = np.tanh(self.W1 @ x + self.b1)
        # FIX: avoid deprecated "array to scalar" conversion
        y_arr = self.W2 @ h + self.b2  # shape (1,)
        y = float(y_arr[0])            # or: y = y_arr.item()
        return y


# ----- Surprise-based gate for mixing shallow & deep -----

class SurpriseGate:
    def __init__(self, alpha: float = 0.1, center: float = 1.0):
        """
        alpha  : EMA coefficient for mean_abs_err
        center : where norm_err = abs_err / mean_abs_err is considered "neutral"
        """
        self.alpha = alpha
        self.center = center
        self.mean_abs_err = None  # will be initialized on first update

    def update_stats(self, abs_err: float):
        abs_err = float(abs(abs_err))
        if self.mean_abs_err is None:
            # First observation: initialize baseline here
            self.mean_abs_err = abs_err if abs_err > 1e-8 else 1.0
        else:
            self.mean_abs_err = (1.0 - self.alpha) * self.mean_abs_err + self.alpha * abs_err

    def compute_gate(self, abs_err: float) -> float:
        """
        Returns g in [0,1].
        - If abs_err is much larger than baseline => g ~ 1 (more deep).
        - If abs_err is much smaller => g ~ 0 (shallow is enough).
        """
        abs_err = float(abs(abs_err))

        if self.mean_abs_err is None:
            # Not enough information yet; stay in the middle
            return 0.5

        norm_err = abs_err / (self.mean_abs_err + 1e-8)  # ~1 when typical
        # Center around 1.0: norm_err > 1 => positive, <1 => negative
        x = norm_err - self.center
        # Logistic squashing
        g = 1.0 / (1.0 + np.exp(-x))
        g = float(np.clip(g, 0.0, 1.0))
        return g


# ----- Full brain wrapper -----

class C7FullBrain:
    def __init__(self, dim_emb_c: int):
        self.shallow = ShallowCore(dim_emb_c)
        self.deep = DeepCore(dim_emb_c, dim_hidden=16)
        self.gate = SurpriseGate(alpha=0.1, center=1.0)

    def forward(self, emb_c: np.ndarray, target: float):
        # Shallow and deep predictions
        y_shallow = self.shallow(emb_c)
        y_deep = self.deep(emb_c)

        # Base error = error of shallow branch w.r.t. target
        base_err = target - y_shallow
        abs_err = abs(base_err)

        # Update gate statistics
        self.gate.update_stats(abs_err)
        g = self.gate.compute_gate(abs_err)

        # Final mixed prediction
        y_final = (1.0 - g) * y_shallow + g * y_deep

        return {
            "y_shallow": y_shallow,
            "y_deep": y_deep,
            "g": g,
            "y_final": y_final,
            "target": target,
            "base_err": base_err,
            "mean_abs_err": self.gate.mean_abs_err,
        }


# ----- Demo / quick test -----

if __name__ == "__main__":
    # Same test Emb-C and target you used before
    emb_c = np.array([2.0, 1.0, 2.0, 3.0, 3.0, 3.0, -1.0, 0.0, 1.0], dtype=float)
    target = 14.0

    brain = C7FullBrain(dim_emb_c=emb_c.size)

    # Run a few times to see how mean_abs_err and g evolve
    for i in range(3):
        out = brain.forward(emb_c, target)
        print(f"Run {i+1}")
        print(f"  y_shallow : {out['y_shallow']}")
        print(f"  y_deep    : {out['y_deep']}")
        print(f"  gate g    : {out['g']}")
        print(f"  y_final   : {out['y_final']}")
        print(f"  target    : {out['target']}")
        print(f"  base_err  : {out['base_err']}")
        print(f"  mean_abs_err: {out['mean_abs_err']}")
        print("-" * 40)