import numpy as np
import os
from collections import deque


# ===============================
#   Simple Frontends (same idea)
# ===============================

class AudioFrontend:
    """
    Very simple audio frontend that transforms 3-dim input
    into a float vector. Just a toy mapping.
    """
    def __call__(self, x):
        x = np.array(x, dtype=float)
        # Normalize into a small range
        return x * 1.0 + np.array([1.0, 0.5, 1.5])


class TextFrontend:
    """
    Simple text frontend: maps 3 integers into 3 floats,
    pretending it's a simple token -> embedding transform.
    """
    def __call__(self, x):
        x = np.array(x, dtype=float)
        # Shift + scale a bit
        base = np.array([3.0, 4.0, 2.0])
        return base + x * 2.0


class ImageFrontend:
    """
    Simple image frontend: maps 3 ints (coarse visual codes)
    into float "embedding".
    """
    def __call__(self, x):
        x = np.array(x, dtype=float)
        return x * 1.0 + np.array([0.5, 1.0, 0.5])


# ============================================
#   Self-Regulation Engine with Horizontal Gate
# ============================================

class SelfRegEngineV72:
    """
    Self-regulation in 3 سطح:

    1) لوکال (اینجا لوکالمون همون خروجی هر آرایه‌ست – scalar)
    2) افقی: بالانس بین A1, A3, A5 (حفظ پهنای‌باند)
    3) عمودی (زمانی): ترکیب با حافظه A7 قبلی، کنترل‌شده با intensity + coherence
    """

    def __init__(self, max_mem_depth=20):
        self.max_mem_depth = max_mem_depth
        self.history_A7 = deque(maxlen=max_mem_depth)
        self.history_intensity = deque(maxlen=max_mem_depth)

    def horizontal_balance(self, array_outputs):
        """
        افقی: اگر واریانس بین آرایه‌ها زیاد باشد،
        کمی آنها را به سمت میانگین می‌کشیم.
        """
        arr = np.array(array_outputs, dtype=float)
        mean = arr.mean()
        var = arr.var()

        # "پهنای‌باند افقی" ~ هرچی var کمتر، coherence بیشتر
        coherence = 1.0 / (1.0 + var)  # بین (۰, ۱]

        # اگر var کوچک است، تقریباً کاری نکن
        var_threshold = 5.0
        if var < var_threshold:
            lambda_h = 0.0
        else:
            # هرچه var بزرگ‌تر، کشش به سمت mean بیشتر
            lambda_h = min(0.6, (var - var_threshold) / (var + 1e-6))

        balanced = arr + lambda_h * (mean - arr)
        return balanced, var, coherence

    def decide_depth_gate(self, intensity, coherence):
        """
        دروازه عمق:

        - شدت بالا => دسترسی کمتر به حافظه
        - coherence پایین => اجازه عمق کمتر
        """
        # شدت: ۰ کم، ۱ زیاد
        # coherence: ۰ کم (var بالا)، ۱ زیاد (var پایین)
        # عمق مجاز ~ آن‌جایی که هم شدت خیلی زیاد نباشد و هم coherence بد نباشد
        depth_permission = coherence * (1.0 - intensity)  # هر دو اثر می‌گذارند
        depth_permission = float(np.clip(depth_permission, 0.0, 1.0))

        # چند قدم حافظه را نگاه کنیم؟
        mem_depth = 1 + int(depth_permission * (self.max_mem_depth - 1))

        # چقدر با حافظه blend کنیم؟
        alpha_max = 0.7  # سقف blend
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
        """
        ورودی:
            arrays_raw : [A1, A3, A5] قبل از بالانس
            intensity  : شدت لحظه‌ای

        خروجی:
            dict شامل:
                - arrays_raw
                - arrays_balanced
                - A7_raw
                - A7_balanced
                - A7_final
                - coherence, var, mem_depth, alpha_smooth, depth_permission
        """
        arr_raw = np.array(arrays_raw, dtype=float)
        A7_raw = float(arr_raw.mean())

        # --- سطح ۲: رگولیت افقی
        arr_balanced, var, coherence = self.horizontal_balance(arr_raw)
        A7_balanced = float(arr_balanced.mean())

        # --- سطح ۳: دروازه عمق + حافظه
        mem_depth, alpha_smooth, depth_permission = self.decide_depth_gate(
            intensity, coherence
        )
        mem_avg_A7 = self.get_memory_avg(mem_depth)

        if mem_avg_A7 is None or alpha_smooth <= 1e-6:
            # حافظه قابل استفاده نیست
            A7_final = A7_balanced
        else:
            A7_final = (1.0 - alpha_smooth) * A7_balanced + alpha_smooth * mem_avg_A7

        # حافظه را با مقدار «قبل از blend» آپدیت می‌کنیم (یا می‌توانستیم بعد از blend هم بکنیم)
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

        # سه آرایه افقی: A1, A3, A5 (هر کدام یک مدل خطی کوچک)
        rng = np.random.default_rng(seed=42)
        self.W_a1 = rng.normal(scale=0.1, size=(input_dim,))
        self.b_a1 = 0.0

        self.W_a3 = rng.normal(scale=0.1, size=(input_dim,))
        self.b_a3 = 0.0

        self.W_a5 = rng.normal(scale=0.1, size=(input_dim,))
        self.b_a5 = 0.0

        # سلف‌رجولیت افقی + عمودی
        self.self_reg = SelfRegEngineV72(max_mem_depth=20)

    def forward_arrays(self, emb_c):
        """
        محاسبه خروجی سه آرایه روی Emb-C
        """
        A1 = float(np.dot(self.W_a1, emb_c) + self.b_a1)
        A3 = float(np.dot(self.W_a3, emb_c) + self.b_a3)
        A5 = float(np.dot(self.W_a5, emb_c) + self.b_a5)
        return A1, A3, A5

    def train_step(self, emb_c_batch, target_batch):
        """
        یک قدم آموزش ساده روی batch از emb_c و target.
        """
        batch_size = emb_c_batch.shape[0]

        # forward
        A1_batch = emb_c_batch @ self.W_a1 + self.b_a1
        A3_batch = emb_c_batch @ self.W_a3 + self.b_a3
        A5_batch = emb_c_batch @ self.W_a5 + self.b_a5

        A7_batch = (A1_batch + A3_batch + A5_batch) / 3.0

        # loss = MSE(A7, target)
        diff = A7_batch - target_batch
        loss = np.mean(diff ** 2)

        # gradients
        dA7 = 2.0 * diff / batch_size  # dL/dA7

        dA1 = dA7 / 3.0
        dA3 = dA7 / 3.0
        dA5 = dA7 / 3.0

        # dL/dW = emb_c^T * dA
        dW_a1 = emb_c_batch.T @ dA1
        dW_a3 = emb_c_batch.T @ dA3
        dW_a5 = emb_c_batch.T @ dA5

        db_a1 = np.sum(dA1)
        db_a3 = np.sum(dA3)
        db_a5 = np.sum(dA5)

        # update
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
        """
        روی یک نمونه، آرایه‌ها را می‌گیرد، self-reg افقی+عمودی را اعمال می‌کند.
        """
        A1, A3, A5 = self.forward_arrays(emb_c)
        reg_info = self.self_reg([A1, A3, A5], intensity)
        return reg_info


# ===========================
#   Utils for training
# ===========================

def compute_intensity(audio_emb, text_emb, image_emb):
    """
    شدت: بر اساس انرژی هر modality و نسبت غالب.
    """
    ea = float(np.linalg.norm(audio_emb))
    et = float(np.linalg.norm(text_emb))
    ei = float(np.linalg.norm(image_emb))
    energies = {"audio": ea, "text": et, "image": ei}

    total = ea + et + ei + 1e-8
    dominant = max(energies, key=energies.get)
    intensity = energies[dominant] / total
    return intensity, energies, dominant


def sample_random_input(rng):
    """
    سه ورودی تصادفی ساده برای audio/text/image
    """
    audio = rng.integers(-1, 4, size=(3,))
    text = rng.integers(-1, 4, size=(3,))
    image = rng.integers(-1, 4, size=(3,))
    return audio, text, image


def main():
    rng = np.random.default_rng(seed=123)

    audio_fe = AudioFrontend()
    text_fe = TextFrontend()
    image_fe = ImageFrontend()

    brain = C7BrainV72(lr=1e-3)

    if brain.load():
        print("Loaded existing C7BrainV7.2 from file.")
    else:
        print("No previous brain found. Creating a NEW C7BrainV7.2.")

    # -------------------
    # Training Phase 7.2
    # -------------------
    print("Starting Phase 7.2 training for 300 steps...")
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

    print("Phase 7.2 training finished.\n")
    brain.save()

    # -------------------
    # Test on one input
    # -------------------
    print("========== TEST AFTER PHASE 7.2 ==========\n")

    # همون ورودی تست قبلی
    audio_in = np.array([2, 1, 2])
    text_in = np.array([3, 3, 3])
    image_in = np.array([-1, 0, 1])

    a_emb = audio_fe(audio_in)
    t_emb = text_fe(text_in)
    i_emb = image_fe(image_in)

    emb_c = np.concatenate([a_emb, t_emb, i_emb])
    target = float(np.sum(emb_c))

    intensity, energies, dominant = compute_intensity(a_emb, t_emb, i_emb)

    print("=== Raw Inputs ===")
    print("Audio Input :", audio_in.tolist())
    print("Text Input  :", text_in.tolist())
    print("Image Input :", image_in.tolist())
    print()

    print("=== Frontend Outputs ===")
    print("AudioFrontend Output :", a_emb.tolist())
    print("TextFrontend Output  :", t_emb.tolist())
    print("ImageFrontend Output :", i_emb.tolist())
    print()

    print("=== Emb-C & Intensity ===")
    print("Emb-C Collapsed Vector:", emb_c.tolist())
    print(f"Intensity             : {intensity:.3f}")
    print("Energies              :", {k: round(v, 3) for k, v in energies.items()})
    print("Dominant modality     :", dominant)
    print()

    # بدون سلف‌رج
    A1_raw, A3_raw, A5_raw = brain.forward_arrays(emb_c)
    A7_raw = (A1_raw + A3_raw + A5_raw) / 3.0

    # با سلف‌رج افقی+عمودی
    reg = brain.run_self_reg(emb_c, intensity)

    print("=== Arrays & A7 (RAW) ===")
    print(f"A1: {A1_raw:.4f}")
    print(f"A3: {A3_raw:.4f}")
    print(f"A5: {A5_raw:.4f}")
    print(f"A7 RAW (mean): {A7_raw:.4f}")
    print(f"Target(sum Emb-C): {target:.4f}")
    print(f"Error (A7_raw - target): {A7_raw - target:.4f}")
    print()

    print("=== Horizontal Coherence & Balancing ===")
    print(f"Var(A1,A3,A5): {reg['var']:.4f}")
    print(f"Coherence    : {reg['coherence']:.4f}")
    print("Arrays (balanced):",
          [float(f"{v:.4f}") for v in reg['arrays_balanced']])
    print(f"A7_balanced  : {reg['A7_balanced']:.4f}")
    print()

    print("=== Depth Gate & Temporal Blend ===")
    print(f"mem_depth       : {reg['mem_depth']}")
    print(f"alpha_smooth    : {reg['alpha_smooth']:.3f}")
    print(f"depth_permission: {reg['depth_permission']:.3f}")
    print()

    print("=== Final A7 With Self-Regulation ===")
    print(f"A7 FINAL (self-regulated): {reg['A7_final']:.4f}")
    print(f"Final error (A7_final - target): {reg['A7_final'] - target:.4f}")
    print("\n=========================================")


if __name__ == "__main__":
    main()