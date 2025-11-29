import numpy as np
import random
from collections import deque

# ================== Phase 6 — C7 with Temporal Memory & Phase Manager ==================

# ----- Utility: random seed for reproducibility (optional) -----
random.seed(42)
np.random.seed(42)


# ----- Simple Frontends (toy but consistent) -----
def audio_frontend(x):
    """
    x: list of 3 ints in range [-1, 3]
    Maps to a small 3D vector. Just a simple linear transform.
    """
    v = np.array(x, dtype=float)
    # center around 0 and scale
    return v * 1.0 + np.array([1.0, 0.5, 1.5])


def text_frontend(x):
    """
    x: list of 3 ints in range [-1, 3]
    Make text more "dominant" (bigger values).
    """
    v = np.array(x, dtype=float)
    return v * 3.0 + np.array([6.0, 7.0, 5.0])


def image_frontend(x):
    """
    x: list of 3 ints in range [-1, 3]
    Intermediate scale.
    """
    v = np.array(x, dtype=float)
    return v * 1.5 + np.array([0.5, 2.0, -0.5])


# ----- Emb-C: collapse 3 modalities into a single 9D vector -----
def build_emb_c(a_out, t_out, i_out):
    return np.concatenate([a_out, t_out, i_out])


# ----- Intensity: simple energy measure from modalities -----
def compute_intensity(a_out, t_out, i_out):
    # Use L2 norms as "energy" and normalize a bit
    audio_e = float(np.linalg.norm(a_out))
    text_e = float(np.linalg.norm(t_out))
    image_e = float(np.linalg.norm(i_out))
    total = audio_e + text_e + image_e + 1e-8
    # intensity in [0, 1]-ish
    intensity = total / (total + 10.0)
    energies = {
        "audio": audio_e,
        "text": text_e,
        "image": image_e,
    }
    return intensity, energies


# ----- C7 Brain (arrays A1, A3, A5) -----
class C7Brain:
    def __init__(self, input_dim=9):
        # 3 arrays, each is a small linear layer
        self.W1 = np.random.randn(input_dim)
        self.W3 = np.random.randn(input_dim)
        self.W5 = np.random.randn(input_dim)

    def forward_arrays(self, emb_c):
        a1 = float(np.dot(self.W1, emb_c))
        a3 = float(np.dot(self.W3, emb_c))
        a5 = float(np.dot(self.W5, emb_c))
        return a1, a3, a5

    def a7_output(self, a1, a3, a5):
        return (a1 + a3 + a5) / 3.0

    def train_step(self, emb_c, target, lr=0.0005):
        """
        Simple MSE loss: (A7 - target)^2
        Gradient with respect to W1, W3, W5.
        """
        a1, a3, a5 = self.forward_arrays(emb_c)
        a7 = self.a7_output(a1, a3, a5)
        error = a7 - target
        loss = error ** 2

        # gradient of a7 w.r.t each Wi is emb_c/3
        grad_common = (2.0 * error) / 3.0 * emb_c

        self.W1 -= lr * grad_common
        self.W3 -= lr * grad_common
        self.W5 -= lr * grad_common

        return loss, a1, a3, a5, a7


# ----- Phase Manager (A, Boundary A→B, B, Boundary B→A) -----
class PhaseManager:
    """
    States:
        0: Phase A
        1: Boundary A→B
        2: Phase B
        3: Boundary B→A
    """

    def __init__(self, initial_state=0, intensity_window=20, threshold_factor=0.15):
        self.state = initial_state
        self.int_history = deque(maxlen=intensity_window)
        self.threshold_factor = threshold_factor

    def update_intensity(self, intensity):
        self.int_history.append(intensity)

    def mean_intensity(self):
        if not self.int_history:
            return 0.0
        return float(np.mean(self.int_history))

    def maybe_transition(self, intensity):
        """
        Only when we are in a boundary state (1 or 3),
        and intensity is sufficiently above mean, we trigger phase change.
        """
        mean_int = self.mean_intensity()
        delta = intensity - mean_int
        trigger = False

        if self.state in (1, 3):
            if delta > self.threshold_factor * (mean_int + 1e-8):
                trigger = True

        if trigger:
            if self.state == 1:
                self.state = 2  # A→B
            elif self.state == 3:
                self.state = 0  # B→A

        return trigger

    def soft_drift(self, intensity):
        """
        Optional: if intensity خیلی پایینه و مدت‌ها تو فاز A یا B هستیم،
        کم‌کم به سمت مرز نزدیک شویم.
        """
        if len(self.int_history) < self.int_history.maxlen:
            return  # هنوز زوده برای drift

        mean_int = self.mean_intensity()

        # اگر شدت خیلی کمتر از میانگینه، سیستم کم‌کم می‌ره سمت مرز
        if self.state == 0 and intensity < 0.9 * mean_int:
            self.state = 1  # نزدیک مرز A→B
        elif self.state == 2 and intensity < 0.9 * mean_int:
            self.state = 3  # نزدیک مرز B→A

    def state_name(self):
        mapping = {
            0: "Phase A",
            1: "Boundary A→B",
            2: "Phase B",
            3: "Boundary B→A",
        }
        return mapping.get(self.state, "Unknown")


# ----- Temporal Memory (کوتاه‌مدت) -----
class TemporalMemory:
    def __init__(self, maxlen=10):
        self.history = deque(maxlen=maxlen)

    def push(self, a7_output, intensity, phase_state):
        self.history.append({
            "a7": a7_output,
            "intensity": intensity,
            "phase": phase_state
        })

    def summary(self):
        if not self.history:
            return {
                "len": 0,
                "avg_a7": None,
                "avg_intensity": None,
            }
        a7_vals = [h["a7"] for h in self.history]
        ints = [h["intensity"] for h in self.history]
        return {
            "len": len(self.history),
            "avg_a7": float(np.mean(a7_vals)),
            "avg_intensity": float(np.mean(ints)),
        }


# ----- Meta-Observer (فقط ناظر، نه کنترل‌گر) -----
class MetaObserver:
    def __init__(self):
        self.last_a7 = None

    def observe(self, a1, a3, a5, a7, target, intensity, phase_state, tmem: TemporalMemory):
        """
        فقط چند شاخص محاسبه و برمی‌گرداند.
        """
        # variance بین آرایه‌ها
        arr = np.array([a1, a3, a5])
        var_arrays = float(np.var(arr))

        error = a7 - target
        abs_error = abs(error)

        # ثبات: آرایه‌ها نزدیک هم + خطا کم
        stability_score = float(np.exp(-var_arrays) * np.exp(-abs_error / 5.0))

        # اگر بین چند تیک اخیر، intensity بالا بوده:
        tmem_summary = tmem.summary()
        recent_int = tmem_summary["avg_intensity"] if tmem_summary["avg_intensity"] is not None else 0.0

        info = {
            "var_arrays": var_arrays,
            "error": error,
            "abs_error": abs_error,
            "stability_score": stability_score,
            "recent_avg_intensity": recent_int,
            "phase_state": phase_state,
        }

        self.last_a7 = a7
        return info


# ================== TRAINING & DEMO ==================

def random_input():
    # سه ورودی در بازه [-1, 3]
    a = [random.randint(-1, 3) for _ in range(3)]
    t = [random.randint(-1, 3) for _ in range(3)]
    i = [random.randint(-1, 3) for _ in range(3)]
    return a, t, i


def main():
    brain = C7Brain(input_dim=9)
    phase_manager = PhaseManager()
    tmem = TemporalMemory(maxlen=10)
    meta = MetaObserver()

    print("Starting Phase 6 training for 300 steps...\n")

    num_steps = 300
    losses = []

    for step in range(1, num_steps + 1):
        audio_in, text_in, image_in = random_input()

        a_out = audio_frontend(audio_in)
        t_out = text_frontend(text_in)
        i_out = image_frontend(image_in)

        emb_c = build_emb_c(a_out, t_out, i_out)
        target = float(np.sum(emb_c))

        intensity, energies = compute_intensity(a_out, t_out, i_out)
        phase_manager.update_intensity(intensity)

        loss, a1, a3, a5, a7 = brain.train_step(emb_c, target)
        losses.append(loss)

        # drift آرام فاز بر اساس شدت
        phase_manager.soft_drift(intensity)
        # تلاش برای ترنزیشن واقعی اگر در مرز هستیم
        trigger = phase_manager.maybe_transition(intensity)

        # حافظهٔ زمانی را آپدیت کن
        tmem.push(a7, intensity, phase_manager.state)

        if step % 50 == 0:
            avg_loss = float(np.mean(losses[-50:]))
            print(f"Step {step}/{num_steps} - avg loss: {avg_loss:.4f} - phase: {phase_manager.state_name()} - trigger: {trigger}")

    print("\nPhase 6 training finished.\n")

    # ======= TEST AFTER TRAINING =======
    print("========== TEST AFTER PHASE 6 TRAINING ==========\n")

    # یک ورودی تست، مثل قبل ولی ثابت
    audio_in = [2, 1, 2]
    text_in = [3, 3, 3]
    image_in = [-1, 0, 1]

    print("=== Raw Inputs ===")
    print("Audio Input :", audio_in)
    print("Text Input  :", text_in)
    print("Image Input :", image_in)
    print()

    a_out = audio_frontend(audio_in)
    t_out = text_frontend(text_in)
    i_out = image_frontend(image_in)

    print("=== Frontend Outputs ===")
    print("AudioFrontend Output :", a_out)
    print("TextFrontend Output  :", t_out)
    print("ImageFrontend Output :", i_out)
    print()

    emb_c = build_emb_c(a_out, t_out, i_out)
    target = float(np.sum(emb_c))

    intensity, energies = compute_intensity(a_out, t_out, i_out)

    print("=== Emb-C & Intensity ===")
    print("Emb-C Collapsed Vector:", emb_c)
    print("Intensity             :", intensity)
    print("Energies              :", energies)
    print()

    # یک عبور از آرایه‌ها
    a1, a3, a5 = brain.forward_arrays(emb_c)
    a7 = brain.a7_output(a1, a3, a5)

    # متا ناظر
    meta_info = meta.observe(
        a1, a3, a5, a7,
        target=target,
        intensity=intensity,
        phase_state=phase_manager.state_name(),
        tmem=tmem
    )

    print("=== Arrays & A7 Output ===")
    print(f"A1: {a1:.4f}")
    print(f"A3: {a3:.4f}")
    print(f"A5: {a5:.4f}")
    print(f"A7 (final): {a7:.4f}")
    print(f"Target(sum Emb-C): {target:.4f}")
    print(f"Error (A7 - target): {a7 - target:.4f}")
    print()

    print("=== Phase Manager State (after training) ===")
    print("Phase:", phase_manager.state_name())
    print()

    print("=== Temporal Memory Summary ===")
    tmem_s = tmem.summary()
    print("History length      :", tmem_s["len"])
    print("Avg A7 in memory    :", tmem_s["avg_a7"])
    print("Avg Intensity memory:", tmem_s["avg_intensity"])
    print()

    print("=== Meta-Observer Info ===")
    for k, v in meta_info.items():
        print(f"{k}: {v}")

    print("\n=============================================")


if __name__ == "__main__":
    main()