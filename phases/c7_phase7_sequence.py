import numpy as np
import os
from collections import deque


# ===============================
#   Simple Frontends (same as 7.2)
# ===============================

class AudioFrontend:
    def __call__(self, x):
        x = np.array(x, dtype=float)
        return x * 1.0 + np.array([1.0, 0.5, 1.5])


class TextFrontend:
    def __call__(self, x):
        x = np.array(x, dtype=float)
        base = np.array([3.0, 4.0, 2.0])
        return base + x * 2.0


class ImageFrontend:
    def __call__(self, x):
        x = np.array(x, dtype=float)
        return x * 1.0 + np.array([0.5, 1.0, 0.5])


# ============================================
#   Self-Regulation Engine with Horizontal Gate
# ============================================

class SelfRegEngineV72:
    """
    سه سطح خودتنظیم‌گری:

    1) محلی: روی خروجی هر آرایه (A1/A3/A5)
    2) افقی: بالانس بین آرایه‌ها برای حفظ «پهنای‌باند افقی»
    3) عمودی/زمانی: blend با حافظه A7 قبلی طبق شدت و coherence
    """

    def __init__(self, max_mem_depth=20):
        self.max_mem_depth = max_mem_depth
        self.history_A7 = deque(maxlen=max_mem_depth)
        self.history_intensity = deque(maxlen=max_mem_depth)

    def horizontal_balance(self, array_outputs):
        """
        اگر واریانس بین آرایه‌ها زیاد باشد، کمی آن‌ها را به سمت میانگین می‌کشیم.
        """
        arr = np.array(array_outputs, dtype=float)
        mean = arr.mean()
        var = arr.var()

        # coherence ~ هرچه var کمتر، هم‌راستایی بیشتر
        coherence = 1.0 / (1.0 + var)

        # آستانه حساس‌تر از نسخه قبلی، که زودتر وارد عمل بشه
        var_threshold = 1.0
        if var < var_threshold:
            lambda_h = 0.0
        else:
            # هرچه var بزرگ‌تر، کشش به سمت mean بیشتر
            lambda_h = min(0.6, (var - var_threshold) / (var + 1e-6))

        balanced = arr + lambda_h * (mean - arr)
        return balanced, var, coherence

    def decide_depth_gate(self, intensity, coherence):
        """
        Gate عمق حافظه:
        - شدت بالا → دسترسی کمتر به حافظه
        - coherence پایین → اجازه عمق کمتر
        """
        depth_permission = coherence * (1.0 - intensity)
        depth_permission = float(np.clip(depth_permission, 0.0, 1.0))

        mem_depth = 1 + int(depth_permission * (self.max_mem_depth - 1))
        alpha_max = 0.7
        alpha_smooth = depth_permission * alpha_max

        return mem_depth, alpha_smooth, depth_permission

    def update_memory(self, A7_value, intensity):
        self.history_A7.append(float(A7_value))
        self.history_intensity.append(float(intensity))

    def get_memory_avg(self, depth):
        if not self.history_A7:
            return None
        depth = min(depth, len(self.history_A7))
        if depth <= 0:
            return None
        vals = list(self.history_A7)[-depth:]
        return float(np.mean(vals))

    def __call__(self, arrays_raw, intensity):
        arr_raw = np.array(arrays_raw, dtype=float)
        A7_raw = float(arr_raw.mean())

        # --- سطح ۲: افقی
        arr_balanced, var, coherence = self.horizontal_balance(arr_raw)
        A7_balanced = float(arr_balanced.mean())

        # --- سطح ۳: عمودی + حافظه
        mem_depth, alpha_smooth, depth_permission = self.decide_depth_gate(
            intensity, coherence
        )
        mem_avg_A7 = self.get_memory_avg(mem_depth)

        if mem_avg_A7 is None or alpha_smooth <= 1e-6:
            A7_final = A7_balanced
        else:
            A7_final = (1.0 - alpha_smooth) * A7_balanced + alpha_smooth * mem_avg_A7

        # حافظه را با مقدار balanced آپدیت می‌کنیم
        self.update_memory(A7_balanced, intensity)

        return {
            "arrays_raw": arr_raw,
            "A7_raw": A7_raw,
            "arrays_balanced": arr_balanced,
            "A7_balanced": A7_balanced,
            "A7_final": A7_final,
            "var": float(var),
            "coherence": float(coherence),
            "mem_depth": int(mem_depth),
            "alpha_smooth": float(alpha_smooth),
            "depth_permission": float(depth_permission),
        }


# ===========================
#   Main C7 Brain v7.2
# ===========================

class C7BrainV72:
    def __init__(self, input_dim=9, lr=1e-3):
        self.input_dim = input_dim
        self.lr = lr

        rng = np.random.default_rng(seed=42)
        self.W_a1 = rng.normal(scale=0.1, size=(input_dim,))
        self.b_a1 = 0.0

        self.W_a3 = rng.normal(scale=0.1, size=(input_dim,))
        self.b_a3 = 0.0

        self.W_a5 = rng.normal(scale=0.1, size=(input_dim,))
        self.b_a5 = 0.0

        self.self_reg = SelfRegEngineV72(max_mem_depth=20)

    def forward_arrays(self, emb_c):
        A1 = float(np.dot(self.W_a1, emb_c) + self.b_a1)
        A3 = float(np.dot(self.W_a3, emb_c) + self.b_a3)
        A5 = float(np.dot(self.W_a5, emb_c) + self.b_a5)
        return A1, A3, A5

    def train_step(self, emb_c_batch, target_batch):
        batch_size = emb_c_batch.shape[0]

        A1_batch = emb_c_batch @ self.W_a1 + self.b_a1
        A3_batch = emb_c_batch @ self.W_a3 + self.b_a3
        A5_batch = emb_c_batch @ self.W_a5 + self.b_a5

        A7_batch = (A1_batch + A3_batch + A5_batch) / 3.0

        diff = A7_batch - target_batch
        loss = np.mean(diff ** 2)

        dA7 = 2.0 * diff / batch_size

        dA1 = dA7 / 3.0
        dA3 = dA7 / 3.0
        dA5 = dA7 / 3.0

        dW_a1 = emb_c_batch.T @ dA1
        dW_a3 = emb_c_batch.T @ dA3
        dW_a5 = emb_c_batch.T @ dA5

        db_a1 = np.sum(dA1)
        db_a3 = np.sum(dA3)
        db_a5 = np.sum(dA5)

        self.W_a1 -= self.lr * dW_a1
        self.b_a1 -= self.lr * db_a1

        self.W_a3 -= self.lr * dW_a3
        self.b_a3 -= self.lr * db_a3

        self.W_a5 -= self.lr * dW_a5
        self.b_a5 -= self.lr * db_a5

        return loss

    def save(self, path="c7_brain_v7_2.npz"):
        np.savez(
            path,
            W_a1=self.W_a1,
            b_a1=self.b_a1,
            W_a3=self.W_a3,
            b_a3=self.b_a3,
            W_a5=self.W_a5,
            b_a5=self.b_a5,
        )

    def load(self, path="c7_brain_v7_2.npz"):
        if not os.path.exists(path):
            return False
        data = np.load(path)
        self.W_a1 = data["W_a1"]
        self.b_a1 = float(data["b_a1"])
        self.W_a3 = data["W_a3"]
        self.b_a3 = float(data["b_a3"])
        self.W_a5 = data["W_a5"]
        self.b_a5 = float(data["b_a5"])
        return True

    def run_self_reg(self, emb_c, intensity):
        A1, A3, A5 = self.forward_arrays(emb_c)
        reg_info = self.self_reg([A1, A3, A5], intensity)
        return reg_info


# ===========================
#   Utils
# ===========================

def compute_intensity(audio_emb, text_emb, image_emb):
    ea = float(np.linalg.norm(audio_emb))
    et = float(np.linalg.norm(text_emb))
    ei = float(np.linalg.norm(image_emb))
    energies = {"audio": ea, "text": et, "image": ei}

    total = ea + et + ei + 1e-8
    dominant = max(energies, key=energies.get)
    intensity = energies[dominant] / total
    return intensity, energies, dominant


def sample_random_input(rng):
    audio = rng.integers(-1, 4, size=(3,))
    text = rng.integers(-1, 4, size=(3,))
    image = rng.integers(-1, 4, size=(3,))
    return audio, text, image


# ===========================
#   MAIN (Sequence Test)
# ===========================

def main():
    rng = np.random.default_rng(seed=123)

    audio_fe = AudioFrontend()
    text_fe = TextFrontend()
    image_fe = ImageFrontend()

    brain = C7BrainV72(lr=1e-3)

    if brain.load():
        print("Loaded existing C7BrainV7.2 from file.")
    else:
        print("No previous brain found. Creating and training NEW C7BrainV7.2...")
        num_steps = 300
        batch_size = 32
        for step in range(1, num_steps + 1):
            emb_batch = []
            target_batch = []
            for _ in range(batch_size):
                audio_in, text_in, image_in = sample_random_input(rng)
                a_emb = audio_fe(audio_in)
                t_emb = text_fe(text_in)
                i_emb = image_fe(image_in)

                emb_c = np.concatenate([a_emb, t_emb, i_emb])
                target = np.sum(emb_c)

                emb_batch.append(emb_c)
                target_batch.append(target)

            emb_batch = np.stack(emb_batch, axis=0)
            target_batch = np.array(target_batch, dtype=float)
            loss = brain.train_step(emb_batch, target_batch)

            if step == 1 or step % 50 == 0:
                print(f"Step {step}/{num_steps} - avg loss: {loss:.4f}")

        brain.save()
        print("Training finished & brain saved.\n")

    # ===========
    # 1) Sequence با ورودی ثابت
    # ===========
    print("\n===== SEQUENCE TEST 1 (FIXED INPUT, 5 STEPS) =====\n")

    audio_in = np.array([2, 1, 2])
    text_in = np.array([3, 3, 3])
    image_in = np.array([-1, 0, 1])

    for i in range(5):
        a_emb = audio_fe(audio_in)
        t_emb = text_fe(text_in)
        i_emb = image_fe(image_in)

        emb_c = np.concatenate([a_emb, t_emb, i_emb])
        target = float(np.sum(emb_c))
        intensity, energies, dominant = compute_intensity(a_emb, t_emb, i_emb)

        reg = brain.run_self_reg(emb_c, intensity)

        print(f"Step {i+1}:")
        print(f"  Intensity       : {intensity:.3f} (dom: {dominant})")
        print(f"  A7_raw          : {reg['A7_raw']:.3f}")
        print(f"  A7_balanced     : {reg['A7_balanced']:.3f}")
        print(f"  A7_final        : {reg['A7_final']:.3f}")
        print(f"  Target          : {target:.3f}")
        print(f"  Error(final)    : {reg['A7_final'] - target:.3f}")
        print(f"  Var(A1,A3,A5)   : {reg['var']:.3f}")
        print(f"  Coherence       : {reg['coherence']:.3f}")
        print(f"  mem_depth       : {reg['mem_depth']}")
        print(f"  alpha_smooth    : {reg['alpha_smooth']:.3f}")
        print(f"  depth_permission: {reg['depth_permission']:.3f}")
        print("-" * 50)

    # ===========
    # 2) Sequence تصادفی
    # ===========
    print("\n===== SEQUENCE TEST 2 (RANDOM INPUTS, 10 STEPS) =====\n")

    for i in range(10):
        audio_in, text_in, image_in = sample_random_input(rng)

        a_emb = audio_fe(audio_in)
        t_emb = text_fe(text_in)
        i_emb = image_fe(image_in)

        emb_c = np.concatenate([a_emb, t_emb, i_emb])
        target = float(np.sum(emb_c))
        intensity, energies, dominant = compute_intensity(a_emb, t_emb, i_emb)

        reg = brain.run_self_reg(emb_c, intensity)

        print(f"Step {i+1}:")
        print(f"  Inputs          : audio={audio_in.tolist()}, text={text_in.tolist()}, image={image_in.tolist()}")
        print(f"  Intensity       : {intensity:.3f} (dom: {dominant})")
        print(f"  A7_raw          : {reg['A7_raw']:.3f}")
        print(f"  A7_balanced     : {reg['A7_balanced']:.3f}")
        print(f"  A7_final        : {reg['A7_final']:.3f}")
        print(f"  Target          : {target:.3f}")
        print(f"  Error(final)    : {reg['A7_final'] - target:.3f}")
        print(f"  Var(A1,A3,A5)   : {reg['var']:.3f}")
        print(f"  Coherence       : {reg['coherence']:.3f}")
        print(f"  mem_depth       : {reg['mem_depth']}")
        print(f"  alpha_smooth    : {reg['alpha_smooth']:.3f}")
        print(f"  depth_permission: {reg['depth_permission']:.3f}")
        print("-" * 80)


if __name__ == "__main__":
    main()