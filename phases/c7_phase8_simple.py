import numpy as np

# ---------- Frontends (ساده) ----------

def audio_frontend(a):
    a = np.array(a, dtype=float)
    return a + np.array([1.0, 0.5, 1.5])

def text_frontend(t):
    t = np.array(t, dtype=float)
    return t * 3.0 + np.array([3.0, 4.0, 2.0])

def image_frontend(i):
    i = np.array(i, dtype=float)
    return i + np.array([1.0, 2.0, 0.0])

def build_emb_c(a, t, i):
    af = audio_frontend(a)
    tf = text_frontend(t)
    imf = image_frontend(i)
    emb = np.concatenate([af, tf, imf])
    return emb, af, tf, imf

def energies_and_intensity(af, tf, imf):
    ea = float(np.linalg.norm(af))
    et = float(np.linalg.norm(tf))
    ei = float(np.linalg.norm(imf))
    total = ea + et + ei + 1e-8
    # شدت فقط یه نرمالایز ساده است
    intensity = (et + ea + ei / 2.0) / (1.5 * total)
    return {"audio": ea, "text": et, "image": ei}, float(intensity)

# ---------- Core Arrays (A1, A3, A5) ----------

class C7Arrays:
    def __init__(self, in_dim=9, lr=0.01):
        self.lr = lr
        self.W1 = np.random.randn(in_dim) * 0.1
        self.W3 = np.random.randn(in_dim) * 0.1
        self.W5 = np.random.randn(in_dim) * 0.1

    def forward(self, emb):
        a1 = float(np.dot(self.W1, emb))
        a3 = float(np.dot(self.W3, emb))
        a5 = float(np.dot(self.W5, emb))
        return a1, a3, a5

    def train_step(self, emb, target):
        a1, a3, a5 = self.forward(emb)
        a7_raw = (a1 + a3 + a5) / 3.0
        err = a7_raw - target
        loss = err * err

        # گرادیان مشترک ساده روی هر سه آرایه
        grad_common = 2.0 * err / 3.0
        self.W1 -= self.lr * grad_common * emb
        self.W3 -= self.lr * grad_common * emb
        self.W5 -= self.lr * grad_common * emb

        return a1, a3, a5, a7_raw, float(loss), float(err)

# ---------- ناظر ساده (SimpleObserver) ----------

class SimpleObserver:
    def __init__(self, max_mem=10):
        self.a7_hist = []
        self.int_hist = []
        self.max_mem = max_mem

    def update_memory(self, a7, intensity):
        self.a7_hist.append(float(a7))
        self.int_hist.append(float(intensity))
        if len(self.a7_hist) > self.max_mem:
            self.a7_hist.pop(0)
            self.int_hist.pop(0)

    def coherence(self, a1, a3, a5):
        arr = np.array([a1, a3, a5], dtype=float)
        var = float(np.var(arr))
        # واریانس کم → کوهرنس بالا
        return 1.0 / (1.0 + var)

    def choose_mode(self, intensity, coh):
        if coh > 0.7:
            return "A"   # پایدار
        elif intensity > 0.7:
            return "C"   # عمق/شدت بالا
        else:
            return "B"   # وسط

    def config(self, intensity, coh):
        mode = self.choose_mode(intensity, coh)
        if mode == "A":
            mem_depth = 3
        elif mode == "B":
            mem_depth = 5
        else:
            mem_depth = 7
        smooth = 0.1 + 0.2 * (1.0 - coh)
        return mode, mem_depth, smooth

    def adjust_a7(self, a7_raw, target, mode, coh):
        """
        فقط یه قدم کوچیک به سمت تارگت.
        شرط: |err_final| <= |err_raw|
        اگر بهتر نشد، همون a7_raw برمی‌گرده.
        """
        err_raw = a7_raw - target

        if mode == "A":
            step = 0.1
        elif mode == "B":
            step = 0.2
        else:  # C
            step = 0.05

        step *= (0.5 + 0.5 * coh)   # کوهرنس بالا → اعتماد بیشتر

        candidate = a7_raw - step * err_raw
        err_cand = candidate - target

        if abs(err_cand) <= abs(err_raw):
            return candidate, err_cand
        else:
            return a7_raw, err_raw

# ---------- Training loop ----------

def random_input_triplet():
    audio = np.random.randint(-1, 4, size=3).tolist()
    text  = np.random.randint(-1, 4, size=3).tolist()
    image = np.random.randint(-1, 4, size=3).tolist()
    return audio, text, image

def train_phase8_simple(steps=300):
    arrays = C7Arrays()
    obs = SimpleObserver()
    losses = []

    for step in range(1, steps + 1):
        a, t, i = random_input_triplet()
        emb, af, tf, imf = build_emb_c(a, t, i)
        target = float(np.sum(emb))

        a1, a3, a5, a7_raw, loss, err_raw = arrays.train_step(emb, target)
        losses.append(loss)

        energies, intensity = energies_and_intensity(af, tf, imf)
        coh = obs.coherence(a1, a3, a5)
        mode, mem_depth, smooth = obs.config(intensity, coh)
        a7_final, err_final = obs.adjust_a7(a7_raw, target, mode, coh)
        obs.update_memory(a7_final, intensity)

        if step % 50 == 0:
            avg_loss = sum(losses[-50:]) / 50.0
            print(
                f"Step {step:3d}/{steps} - loss:{avg_loss:.4f} "
                f"mode:{mode} coh:{coh:.2f} int:{intensity:.2f} "
                f"err_raw:{err_raw:.2f} err_final:{err_final:.2f}"
            )

    print("\nPhase 8 SIMPLE training finished.\n")
    return arrays, obs

def test_phase8_simple(arrays, obs):
    print("========== TEST AFTER PHASE 8 SIMPLE ==========\n")

    audio = [2, 1, 2]
    text  = [3, 3, 3]
    image = [-1, 0, 1]

    emb, af, tf, imf = build_emb_c(audio, text, image)
    target = float(np.sum(emb))
    a1, a3, a5 = arrays.forward(emb)
    a7_raw = (a1 + a3 + a5) / 3.0

    energies, intensity = energies_and_intensity(af, tf, imf)
    coh = obs.coherence(a1, a3, a5)
    mode, mem_depth, smooth = obs.config(intensity, coh)
    a7_final, err_final = obs.adjust_a7(a7_raw, target, mode, coh)

    print("=== Raw Inputs ===")
    print("Audio Input :", audio)
    print("Text Input  :", text)
    print("Image Input :", image)

    print("\n=== Frontend Outputs ===")
    print("AudioFrontend Output :", af)
    print("TextFrontend Output  :", tf)
    print("ImageFrontend Output :", imf)

    print("\n=== Emb-C & Intensity ===")
    print("Emb-C Collapsed Vector:", emb)
    print("Target(sum Emb-C):", f"{target:.4f}")
    print("Intensity             :", f"{intensity:.3f}")
    print("Energies              :", energies)

    print("\n=== Arrays & A7 ===")
    print("A1      :", f"{a1:.4f}")
    print("A3      :", f"{a3:.4f}")
    print("A5      :", f"{a5:.4f}")
    print("A7_raw  :", f"{a7_raw:.4f}")
    print("Mode    :", mode)
    print("Coherence:", f"{coh:.3f}")

    print("\n=== Self-Regulated Output ===")
    print("A7_final:", f"{a7_final:.4f}")
    print("Error_final (A7_final - target):", f"{err_final:.4f}")

    print("\n================================")

if __name__ == "__main__":
    arrays, obs = train_phase8_simple(steps=300)
    test_phase8_simple(arrays, obs)