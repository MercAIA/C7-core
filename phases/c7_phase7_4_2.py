import numpy as np
import json
import os

# ============================================================
#  Phase 7.5 – Architecture C + 3 Modes + Self-Reg + Learning
# ============================================================

class C7BrainV75:
    def __init__(self):
        # weights for three arrays (linear)
        self.W1 = np.random.randn(9, 16) * 0.05
        self.W3 = np.random.randn(9, 16) * 0.05
        self.W5 = np.random.randn(9, 16) * 0.05

        # حافظه برای self-reg
        self.mem_A7 = []
        self.mem_intensity = []

    # ----------------- ذخیره / لود -----------------
    def save(self, path="c7_brain_v75.json"):
        data = {
            "W1": self.W1.tolist(),
            "W3": self.W3.tolist(),
            "W5": self.W5.tolist()
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load(path="c7_brain_v75.json"):
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        brain = C7BrainV75()
        brain.W1 = np.array(data["W1"])
        brain.W3 = np.array(data["W3"])
        brain.W5 = np.array(data["W5"])
        return brain

    # ----------------- Frontends -----------------
    def audio_front(self, a):
        # همون چیزی که در ۷.۴.۱ داشتی
        return np.array([a[0] + 1.0, a[1] + 0.5, a[2] + 1.5])

    def text_front(self, t):
        return np.array([t[0] * 3.0, t[1] * 3.0, t[2] * 3.0 + 1.0])

    def image_front(self, i):
        return np.array([i[0] * 0.5, i[1] + 1.0, i[2] * 0.5])

    # ----------------- Coherence -----------------
    def coherence(self, a1, a3, a5):
        v = np.var([a1, a3, a5])
        return 1.0 / (1.0 + v)

    # ----------------- Mode Selector -----------------
    def select_mode(self, intensity, coherence, error_raw):
        # حالت عمیق C فقط وقتی:
        # شدت زیاد + خطای خام بزرگ + coherence پایین
        if intensity > 0.75 and abs(error_raw) > 5.0 and coherence < 0.40:
            return "C"

        # حالت A وقتی پایدار و هماهنگ
        if intensity < 0.45 and coherence > 0.60:
            return "A"

        # بقیه‌ی موارد: B
        return "B"

    def get_mode_params(self, mode):
        if mode == "A":
            return dict(gain=0.90, mem_depth=2, smooth=0.06, lr=5e-5)
        if mode == "B":
            return dict(gain=1.00, mem_depth=5, smooth=0.12, lr=1e-4)
        if mode == "C":
            return dict(gain=1.10, mem_depth=8, smooth=0.20, lr=8e-5)

    # ----------------- Forward برای TRAIN (با گرادیان) -----------------
    def forward_train(self, audio, text, image):
        # 1) frontends
        A = self.audio_front(audio)
        T = self.text_front(text)
        I = self.image_front(image)

        emb = np.concatenate([A, T, I])  # شکل (9,)

        # normها / intensity
        energies = {
            "audio": np.linalg.norm(A),
            "text": np.linalg.norm(T),
            "image": np.linalg.norm(I)
        }
        intensity = np.linalg.norm(emb) / 40.0

        # 2) خروجی سه آرایه
        ones_vec = np.ones(16)
        a1 = float(emb @ self.W1 @ ones_vec)
        a3 = float(emb @ self.W3 @ ones_vec)
        a5 = float(emb @ self.W5 @ ones_vec)

        a7_raw = (a1 + a3 + a5) / 3.0
        coh = self.coherence(a1, a3, a5)

        target = float(np.sum(emb))
        error_raw = a7_raw - target

        # انتخاب مود بر اساس intensity / coherence / error_raw
        mode = self.select_mode(intensity, coh, error_raw)
        params = self.get_mode_params(mode)

        # loss ساده MSE روی a7_raw
        loss = error_raw ** 2

        # ----------------- گرادیان خیلی ساده -----------------
        # da1/dW1 = emb ⊗ ones_vec
        dA = np.outer(emb, ones_vec)  # (9,16)
        # da7_raw/dak = 1/3 برای هر آریه
        factor = 2.0 * error_raw / 3.0

        dW1 = factor * dA
        dW3 = factor * dA
        dW5 = factor * dA

        lr = params["lr"]
        self.W1 -= lr * dW1
        self.W3 -= lr * dW3
        self.W5 -= lr * dW5

        return {
            "loss": float(loss),
            "mode": mode,
            "coherence": float(coh),
            "intensity": float(intensity),
            "error_raw": float(error_raw)
        }

    # ----------------- Forward برای EVAL (با self-reg) -----------------
    def forward_eval(self, audio, text, image):
        A = self.audio_front(audio)
        T = self.text_front(text)
        I = self.image_front(image)

        emb = np.concatenate([A, T, I])

        energies = {
            "audio": np.linalg.norm(A),
            "text": np.linalg.norm(T),
            "image": np.linalg.norm(I)
        }
        intensity = np.linalg.norm(emb) / 40.0

        ones_vec = np.ones(16)
        a1 = float(emb @ self.W1 @ ones_vec)
        a3 = float(emb @ self.W3 @ ones_vec)
        a5 = float(emb @ self.W5 @ ones_vec)
        a7_raw = (a1 + a3 + a5) / 3.0

        coh = self.coherence(a1, a3, a5)
        target = float(np.sum(emb))
        error_raw = a7_raw - target

        # انتخاب مود، ولی اینجا فقط برای پارامترهای self-reg
        mode = self.select_mode(intensity, coh, error_raw)
        params = self.get_mode_params(mode)

        gain = params["gain"]
        mem_depth = params["mem_depth"]
        smooth = params["smooth"]

        # scale
        a1_scaled = a1 * gain
        a3_scaled = a3 * gain
        a5_scaled = a5 * gain
        a7_scaled = (a1_scaled + a3_scaled + a5_scaled) / 3.0

        # حافظه
        self.mem_A7.append(a7_scaled)
        self.mem_intensity.append(intensity)
        if len(self.mem_A7) > 50:
            self.mem_A7.pop(0)
            self.mem_intensity.pop(0)

        # blend با حافظه
        if len(self.mem_A7) >= mem_depth:
            mem_avg = np.mean(self.mem_A7[-mem_depth:])
        else:
            mem_avg = a7_scaled

        a7_final = smooth * mem_avg + (1.0 - smooth) * a7_scaled
        error_final = a7_final - target

        return {
            "A": A.tolist(),
            "T": T.tolist(),
            "I": I.tolist(),
            "emb": emb.tolist(),
            "intensity": float(intensity),
            "energies": {
                "audio": float(energies["audio"]),
                "text": float(energies["text"]),
                "image": float(energies["image"])
            },
            "mode": mode,
            "coherence": float(coh),
            "a1": float(a1_scaled),
            "a3": float(a3_scaled),
            "a5": float(a5_scaled),
            "a7_raw": float(a7_raw),
            "a7_final": float(a7_final),
            "target": float(target),
            "error_raw": float(error_raw),
            "error_final": float(error_final),
            "params": params
        }


# ----------------- Training Loop -----------------

def train_phase7_5():
    brain = C7BrainV75.load()
    if brain is None:
        print("No previous brain found. Creating NEW C7BrainV75.")
        brain = C7BrainV75()
    else:
        print("Loaded existing C7BrainV75 from file.")

    print("Starting Phase 7.5 training for 300 steps...")

    for step in range(300):
        audio = np.random.randint(-1, 3, 3)
        text  = np.random.randint(-1, 4, 3)
        image = np.random.randint(-1, 3, 3)

        out = brain.forward_train(audio, text, image)

        if (step + 1) % 50 == 0:
            print(
                f"Step {step+1}/300 - loss: {out['loss']:.4f} "
                f"- mode:{out['mode']} coh:{out['coherence']:.2f} "
                f"int:{out['intensity']:.2f} err_raw:{out['error_raw']:.2f}"
            )

    brain.save()
    print("Phase 7.5 training finished. Brain saved.\n")
    return brain


# ----------------- Main -----------------

if __name__ == "__main__":
    brain = train_phase7_5()

    # تست روی همان ورودی مرجع
    test_audio = [2, 1, 2]
    test_text  = [3, 3, 3]
    test_image = [-1, 0, 1]

    out = brain.forward_eval(test_audio, test_text, test_image)

    print("========== TEST AFTER PHASE 7.5 ==========")
    for k, v in out.items():
        print(k, ":", v)