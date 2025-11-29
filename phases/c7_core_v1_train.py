import numpy as np
import math

class CoreBrain:
    def __init__(self, input_dim=9, hidden_dim=16, lr=1e-3):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        rng = np.random.default_rng(42)

        # --- Shallow head ---
        self.Ws1 = rng.normal(scale=0.1, size=(hidden_dim, input_dim))
        self.bs1 = np.zeros(hidden_dim)
        self.Ws2 = rng.normal(scale=0.1, size=(hidden_dim,))
        self.bs2 = 0.0

        # --- Deep head ---
        self.Wd1 = rng.normal(scale=0.1, size=(hidden_dim, input_dim))
        self.bd1 = np.zeros(hidden_dim)
        self.Wd2 = rng.normal(scale=0.1, size=(hidden_dim,))
        self.bd2 = 0.0

        # --- Gate (ثابت، فقط برای دمو) ---
        # weights برای [norm_err, intensity, 1]
        self.g_w = np.array([-1.0, 2.0, 0.5])
        self.g_b = 0.0

        # EMA خطای شلو برای نرمال‌سازی
        self.base_err_ema = 1.0

    def _forward_head(self, x, W1, b1, W2, b2):
        z = W1 @ x + b1          # (hidden,)
        h = np.tanh(z)           # nonlinearity
        y = float(W2 @ h + b2)   # scalar
        return y, h

    def _gate(self, norm_err, intensity):
        z = self.g_w[0] * norm_err + self.g_w[1] * intensity + self.g_w[2] * 1.0 + self.g_b
        g = 1.0 / (1.0 + math.exp(-z))
        return g

    def forward(self, x, target=None, update_state=True):
        x = np.asarray(x, dtype=float)

        # دو هد
        y_s, h_s = self._forward_head(x, self.Ws1, self.bs1, self.Ws2, self.bs2)
        y_d, h_d = self._forward_head(x, self.Wd1, self.bd1, self.Wd2, self.bd2)

        # intensity (یه تقریب ساده)
        audio = x[0:3]
        text  = x[3:6]
        image = x[6:9]
        energy = np.sqrt(audio @ audio + text @ text + image @ image)
        intensity = float(energy / (1.0 + energy))

        # خطای شلو برای گیت
        if target is not None:
            base_err = abs(y_s - target)
            if update_state:
                self.base_err_ema = 0.9 * self.base_err_ema + 0.1 * base_err
            norm_err = base_err / (self.base_err_ema + 1e-6)
        else:
            base_err = 0.0
            norm_err = 0.0

        g = self._gate(norm_err, intensity)
        y_final = (1.0 - g) * y_s + g * y_d

        out = {
            "x": x,
            "y_s": y_s,
            "y_d": y_d,
            "y_final": y_final,
            "h_s": h_s,
            "h_d": h_d,
            "intensity": intensity,
            "gate": g,
            "base_err": base_err,
            "norm_err": norm_err,
        }

        if target is not None:
            out["target"] = target
            out["loss"] = 0.5 * (y_final - target) ** 2

        return out

    def step(self, x, target):
        """
        یک گام آموزش با backprop روی شلو و دیپ.
        گیت فعلاً ثابت فرض شده (روی g گرادیان نمی‌گیریم).
        """
        f = self.forward(x, target, update_state=True)

        x      = f["x"]
        y_s    = f["y_s"]
        y_d    = f["y_d"]
        y_final= f["y_final"]
        h_s    = f["h_s"]
        h_d    = f["h_d"]
        g      = f["gate"]
        target = f["target"]

        # dL/dy_final
        dL_dy = y_final - target

        # گیت ثابت → فقط نسبت اثر شلو/دیپ
        dL_dys = dL_dy * (1.0 - g)
        dL_dyd = dL_dy * g

        # --- شلو ---
        # y_s = Ws2 @ h_s + bs2
        dL_dWs2 = dL_dys * h_s            # (hidden,)
        dL_dbs2 = dL_dys

        # h_s = tanh(z_s), z_s = Ws1 @ x + bs1
        dL_dh_s = dL_dys * self.Ws2       # (hidden,)
        dz_s = dL_dh_s * (1.0 - h_s ** 2) # (hidden,)
        dL_dWs1 = dz_s[:, None] @ x[None, :]  # (hidden, input)
        dL_dbs1 = dz_s

        # --- دیپ ---
        dL_dWd2 = dL_dyd * h_d
        dL_dbd2 = dL_dyd

        dL_dh_d = dL_dyd * self.Wd2
        dz_d = dL_dh_d * (1.0 - h_d ** 2)
        dL_dWd1 = dz_d[:, None] @ x[None, :]
        dL_dbd1 = dz_d

        # آپدیت پارامترها
        self.Ws2 -= self.lr * dL_dWs2
        self.bs2 -= self.lr * dL_dbs2
        self.Ws1 -= self.lr * dL_dWs1
        self.bs1 -= self.lr * dL_dbs1

        self.Wd2 -= self.lr * dL_dWd2
        self.bd2 -= self.lr * dL_dbd2
        self.Wd1 -= self.lr * dL_dWd1
        self.bd1 -= self.lr * dL_dbd1

        return float(f["loss"])

    def train(self, steps=1000):
        rng = np.random.default_rng(0)
        for step in range(1, steps + 1):
            # یک Emb-C تصادفی (شبیه چیزی که خودمون استفاده کردیم)
            x = rng.integers(-1, 4, size=(self.input_dim,))
            target = float(x.sum())

            loss = self.step(x, target)

            if step % 100 == 0:
                print(
                    f"Step {step:4d} | loss:{loss:7.3f}  "
                    f"base_err_ema:{self.base_err_ema:5.3f}"
                )


if __name__ == "__main__":
    brain = CoreBrain()

    fixed_x = [2, 1, 2, 3, 3, 3, -1, 0, 1]
    fixed_target = 14.0

    print("=== BEFORE TRAINING ===")
    res0 = brain.forward(fixed_x, target=fixed_target)
    print(f"y_shallow : {res0['y_s']:.3f}")
    print(f"y_deep    : {res0['y_d']:.3f}")
    print(f"y_final   : {res0['y_final']:.3f}")
    print(f"loss      : {res0['loss']:.3f}")
    print("-" * 40)

    brain.train(steps=1000)

    print("=== AFTER TRAINING ===")
    res1 = brain.forward(fixed_x, target=fixed_target)
    print(f"y_shallow : {res1['y_s']:.3f}")
    print(f"y_deep    : {res1['y_d']:.3f}")
    print(f"y_final   : {res1['y_final']:.3f}")
    print(f"loss      : {res1['loss']:.3f}")