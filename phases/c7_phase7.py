import numpy as np
import os
import pickle

# ============================================
# C7 – Phase 7
# Self-Regulation Engine + Temporal Memory
# ============================================

# ---------- Frontends (همون 3 ماژول ورودی) ----------

def audio_frontend(x):
    x = np.array(x, dtype=float)
    # خروجی کمی نرمال‌تر / شبیه فازهای قبلی
    return np.array([
        x[0] + 1.0,
        0.5 * (x[0] + x[1]),
        x[2] + 1.5
    ])

def text_frontend(x):
    x = np.array(x, dtype=float)
    # متن رو قوی‌تر می‌کنیم (مثل قبل که انرژی متن غالب بود)
    return np.array([
        3.0 + 2.0 * x[0] + x[1],
        4.0 + 2.0 * x[1] + x[2],
        2.0 + 2.0 * x[2] + x[0]
    ])

def image_frontend(x):
    x = np.array(x, dtype=float)
    # تصویر نسبتاً ملایم‌تر
    return np.array([
        0.5 * x[0] + 1.0,
        x[1] + 1.0,
        0.5 * x[2]
    ])


# ---------- Emb-C و شدت (Intensity) ----------

def build_emb_c(a_emb, t_emb, i_emb):
    return np.concatenate([a_emb, t_emb, i_emb])

def compute_energies(a_emb, t_emb, i_emb):
    e_audio = float(np.linalg.norm(a_emb))
    e_text  = float(np.linalg.norm(t_emb))
    e_image = float(np.linalg.norm(i_emb))
    total   = e_audio + e_text + e_image + 1e-8
    return {
        "audio": e_audio,
        "text":  e_text,
        "image": e_image,
        "total": total
    }

def compute_intensity(energies):
    # شدت = نسبت انرژی مد غالب به مجموع
    dominant_val = max(energies["audio"], energies["text"], energies["image"])
    return dominant_val / (energies["total"] + 1e-8)


def dominant_modality(energies):
    vals = {
        "audio": energies["audio"],
        "text":  energies["text"],
        "image": energies["image"]
    }
    return max(vals, key=vals.get)


# ---------- مغز C7 نسخه 7 (سه آرایه + A7) ----------

class C7BrainV7:
    def __init__(self, dim=9):
        # وزن‌های سه آرایه (A1, A3, A5) + bias
        self.dim = dim
        self.w1 = np.random.randn(dim) * 0.1
        self.w3 = np.random.randn(dim) * 0.1
        self.w5 = np.random.randn(dim) * 0.1
        self.b1 = 0.0
        self.b3 = 0.0
        self.b5 = 0.0

    def forward_arrays(self, emb_c):
        # emb_c: وکتور 9 بعدی
        y1 = float(np.dot(self.w1, emb_c) + self.b1)
        y3 = float(np.dot(self.w3, emb_c) + self.b3)
        y5 = float(np.dot(self.w5, emb_c) + self.b5)
        return y1, y3, y5

    def forward(self, emb_c):
        y1, y3, y5 = self.forward_arrays(emb_c)
        a7_raw = (y1 + y3 + y5) / 3.0
        return y1, y3, y5, a7_raw

    def train_step(self, emb_c_batch, targets, lr=1e-3):
        """
        emb_c_batch: [batch, dim]
        targets: [batch]
        """
        batch_size = emb_c_batch.shape[0]

        # گرادیان‌ها
        gw1 = np.zeros_like(self.w1)
        gw3 = np.zeros_like(self.w3)
        gw5 = np.zeros_like(self.w5)
        gb1 = 0.0
        gb3 = 0.0
        gb5 = 0.0

        losses = []

        for i in range(batch_size):
            x = emb_c_batch[i]
            target = targets[i]

            y1, y3, y5 = self.forward_arrays(x)
            a7 = (y1 + y3 + y5) / 3.0
            err = a7 - target
            loss = err * err
            losses.append(loss)

            # dL/d(a7) = 2 * err
            dL_da7 = 2.0 * err

            # چون a7 = (y1 + y3 + y5) / 3
            dL_dy1 = dL_da7 * (1.0 / 3.0)
            dL_dy3 = dL_da7 * (1.0 / 3.0)
            dL_dy5 = dL_da7 * (1.0 / 3.0)

            gw1 += dL_dy1 * x
            gw3 += dL_dy3 * x
            gw5 += dL_dy5 * x
            gb1 += dL_dy1
            gb3 += dL_dy3
            gb5 += dL_dy5

        # آپدیت پارامترها
        gw1 /= batch_size
        gw3 /= batch_size
        gw5 /= batch_size
        gb1 /= batch_size
        gb3 /= batch_size
        gb5 /= batch_size

        self.w1 -= lr * gw1
        self.w3 -= lr * gw3
        self.w5 -= lr * gw5
        self.b1 -= lr * gb1
        self.b3 -= lr * gb3
        self.b5 -= lr * gb5

        return float(np.mean(losses))


# ---------- حافظه زمانی (Temporal Memory) ----------

class TemporalMemory:
    def __init__(self, max_len=20):
        self.max_len = max_len
        self.history = []  # list of dicts: {"A7":..., "intensity":...}

    def add(self, A7, intensity):
        self.history.append({"A7": float(A7), "intensity": float(intensity)})
        if len(self.history) > self.max_len:
            self.history.pop(0)

    def avg_A7(self, depth=None):
        if not self.history:
            return 0.0
        if depth is None or depth <= 0 or depth > len(self.history):
            data = self.history
        else:
            data = self.history[-depth:]
        return float(np.mean([h["A7"] for h in data]))

    def avg_intensity(self, depth=None):
        if not self.history:
            return 0.0
        if depth is None or depth <= 0 or depth > len(self.history):
            data = self.history
        else:
            data = self.history[-depth:]
        return float(np.mean([h["intensity"] for h in data]))

    def length(self):
        return len(self.history)


# ---------- Self-Regulation Engine (فاز ۷) ----------

class SelfRegEngine:
    """
    ورودی:
      - intensity        : شدت فعلی
      - error            : A7_raw - target
      - var_arrays       : واریانس بین A1,A3,A5
      - memory           : TemporalMemory

    خروجی:
      - gain             : اسکالر برای مقیاس‌کردن آرایه‌ها
      - mem_depth        : عمق استفاده از حافظه
      - alpha_smooth     : ضریب ترکیب با میانگین حافظه
    """

    def __init__(self, max_mem_depth=20):
        self.max_mem_depth = max_mem_depth

    def __call__(self, intensity, error, var_arrays, memory: TemporalMemory):
        intensity = float(intensity)
        abs_error = abs(float(error))
        var_arrays = float(var_arrays)

        # --- 1) تنظیم gain ---
        # شدت بالا → کمی کاهش gain
        gain = 1.0
        if intensity > 0.8:
            gain -= 0.2
        elif intensity < 0.4 and abs_error < 1.0:
            # شدت پایین + خطای کم → اجازه افزایش gain
            gain += 0.1

        # اگر نوسان آرایه‌ها زیاد باشد → کمی کاهش gain
        if var_arrays > 50.0:
            gain -= 0.1

        # کلیپ
        gain = max(0.7, min(1.3, gain))

        # --- 2) عمق حافظه ---
        # شدت بالا → حافظه کم‌تر
        # شدت پایین → حافظه عمیق‌تر
        mem_depth = int(1 + (1.0 - intensity) * (self.max_mem_depth - 1))
        if mem_depth < 1:
            mem_depth = 1
        if mem_depth > self.max_mem_depth:
            mem_depth = self.max_mem_depth

        # --- 3) alpha_smooth (ترکیب A7 با میانگین حافظه) ---
        # شدت بالا → وابستگی کمتر به حافظه
        # شدت پایین → وابستگی بیشتر
        alpha_smooth = (1.0 - intensity) * 0.7  # حداکثر 0.7

        return {
            "gain": gain,
            "mem_depth": mem_depth,
            "alpha_smooth": alpha_smooth
        }


# ---------- ذخیره/لود مغز (Persistent) ----------

BRAIN_PATH = "c7_brain_v7.pkl"

def load_or_create_brain(dim=9):
    if os.path.exists(BRAIN_PATH):
        with open(BRAIN_PATH, "rb") as f:
            brain = pickle.load(f)
        print("Loaded existing C7BrainV7 from file.")
    else:
        brain = C7BrainV7(dim=dim)
        print("No previous brain found. Creating a NEW C7BrainV7.")
    return brain

def save_brain(brain):
    with open(BRAIN_PATH, "wb") as f:
        pickle.dump(brain, f)
    print("Brain weights saved.")


# ---------- Training Loop Phase 7 ----------

def random_input_triplet():
    # ورودی تصادفی در بازه‌ای شبیه قبل
    a = np.random.randint(-1, 4, size=3)  # audio
    t = np.random.randint(-1, 4, size=3)  # text
    i = np.random.randint(-1, 4, size=3)  # image
    return a, t, i

def train_phase7(brain, steps=300, batch_size=16, lr=1e-3):
    print("Starting Phase 7 training for {} steps...".format(steps))

    for step in range(1, steps + 1):
        emb_batch = []
        targets = []

        for _ in range(batch_size):
            a_in, t_in, i_in = random_input_triplet()

            a_emb = audio_frontend(a_in)
            t_emb = text_frontend(t_in)
            i_emb = image_frontend(i_in)

            emb_c = build_emb_c(a_emb, t_emb, i_emb)
            target = float(np.sum(emb_c))

            emb_batch.append(emb_c)
            targets.append(target)

        emb_batch = np.stack(emb_batch, axis=0)
        targets = np.array(targets, dtype=float)

        loss = brain.train_step(emb_batch, targets, lr=lr)

        if step % 50 == 0 or step == 1:
            print(f"Step {step}/{steps} - avg loss: {loss:.4f}")

    print("Phase 7 training finished.\n")


# ---------- Test / Demo Phase 7 ----------

def test_phase7(brain):
    print("========== TEST AFTER PHASE 7 ==========\n")

    # همان ورودی تست کلاسیک ما
    audio_input = np.array([2, 1, 2])
    text_input  = np.array([3, 3, 3])
    image_input = np.array([-1, 0, 1])

    print("=== Raw Inputs ===")
    print("Audio Input :", audio_input.tolist())
    print("Text Input  :", text_input.tolist())
    print("Image Input :", image_input.tolist())
    print()

    # Frontends
    a_emb = audio_frontend(audio_input)
    t_emb = text_frontend(text_input)
    i_emb = image_frontend(image_input)

    print("=== Frontend Outputs ===")
    print("AudioFrontend Output :", a_emb.tolist())
    print("TextFrontend Output  :", t_emb.tolist())
    print("ImageFrontend Output :", i_emb.tolist())
    print()

    # Emb-C + Energies + Intensity
    emb_c = build_emb_c(a_emb, t_emb, i_emb)
    energies = compute_energies(a_emb, t_emb, i_emb)
    intensity = compute_intensity(energies)
    dom = dominant_modality(energies)

    print("=== Emb-C & Intensity ===")
    print("Emb-C Collapsed Vector:", emb_c.tolist())
    print(f"Intensity             : {intensity:.3f}")
    print("Energies              :", {
        "audio": round(energies["audio"], 3),
        "text":  round(energies["text"], 3),
        "image": round(energies["image"], 3),
    })
    print(f"Dominant modality     : {dom}")
    print()

    # Forward raw
    y1, y3, y5, a7_raw = brain.forward(emb_c)
    target = float(np.sum(emb_c))
    error = a7_raw - target
    arrays = np.array([y1, y3, y5], dtype=float)
    var_arrays = float(np.var(arrays))

    print("=== Arrays & A7 (RAW) ===")
    print(f"A1: {y1:.4f}")
    print(f"A3: {y3:.4f}")
    print(f"A5: {y5:.4f}")
    print(f"A7 RAW (mean): {a7_raw:.4f}")
    print(f"Target(sum Emb-C): {target:.4f}")
    print(f"Error (A7_raw - target): {error:.4f}")
    print(f"Var(A1,A3,A5): {var_arrays:.4f}")
    print()

    # Temporal Memory + Self-Reg
    memory = TemporalMemory(max_len=20)
    reg = SelfRegEngine(max_mem_depth=20)

    # فرض می‌کنیم چند بار قبلاً سیستم کار کرده؛ یه history مصنوعی می‌سازیم:
    for _ in range(10):
        fake_A7 = a7_raw + np.random.randn() * 0.5
        fake_int = intensity + np.random.randn() * 0.05
        memory.add(fake_A7, fake_int)

    mem_avg_A7_before = memory.avg_A7()
    mem_avg_int_before = memory.avg_intensity()

    regs = reg(
        intensity=intensity,
        error=error,
        var_arrays=var_arrays,
        memory=memory
    )

    gain = regs["gain"]
    mem_depth = regs["mem_depth"]
    alpha_smooth = regs["alpha_smooth"]

    print("=== Self-Regulation Decision ===")
    print(f"gain         : {gain:.3f}")
    print(f"mem_depth    : {mem_depth}")
    print(f"alpha_smooth : {alpha_smooth:.3f}")
    print(f"Memory avg A7 (before): {mem_avg_A7_before:.4f}")
    print(f"Memory avg Intensity  : {mem_avg_int_before:.3f}")
    print()

    # اعمال gain روی آرایه‌ها
    arrays_scaled = arrays * gain
    a7_scaled = float(np.mean(arrays_scaled))

    # استفاده از حافظه با عمق انتخاب‌شده
    mem_avg_A7_depth = memory.avg_A7(depth=mem_depth)

    # ترکیب A7 با حافظه
    a7_final = (1.0 - alpha_smooth) * a7_scaled + alpha_smooth * mem_avg_A7_depth

    print("=== Arrays After Gain (SCALED) ===")
    print(f"A1_scaled: {arrays_scaled[0]:.4f}")
    print(f"A3_scaled: {arrays_scaled[1]:.4f}")
    print(f"A5_scaled: {arrays_scaled[2]:.4f}")
    print(f"A7_scaled (mean): {a7_scaled:.4f}")
    print()

    print("=== Final A7 With Self-Regulation ===")
    print(f"Memory avg A7 (depth={mem_depth}): {mem_avg_A7_depth:.4f}")
    print(f"A7 FINAL (self-regulated): {a7_final:.4f}")
    print(f"Final error (A7_final - target): {a7_final - target:.4f}")
    print("\n=========================================\n")


# ---------- main ----------

def main():
    brain = load_or_create_brain(dim=9)
    train_phase7(brain, steps=300, batch_size=16, lr=1e-3)
    save_brain(brain)
    test_phase7(brain)

if __name__ == "__main__":
    main()