import numpy as np
import os
import pickle

BRAIN_PATH = "c7_brain_v9.pkl"

rng = np.random.default_rng()

def audio_frontend(x):
    x = np.asarray(x, dtype=float)
    # simple affine transform
    return x + np.array([1.0, 0.5, 1.0])

def text_frontend(x):
    x = np.asarray(x, dtype=float)
    # scale up to dominate semantics
    return 4.0 * x

def image_frontend(x):
    x = np.asarray(x, dtype=float)
    # mild shift & scale
    return 1.5 * x + np.array([0.0, 1.0, 0.5])

def compute_energies(a, t, i):
    ea = float(np.linalg.norm(a) + 1e-8)
    et = float(np.linalg.norm(t) + 1e-8)
    ei = float(np.linalg.norm(i) + 1e-8)
    return {
        "audio": ea,
        "text": et,
        "image": ei,
    }

def compute_intensity(energies):
    total = energies["audio"] + energies["text"] + energies["image"]
    return energies["text"] / total

def coherence_from_arrays(a1, a3, a5):
    vals = np.array([a1, a3, a5], dtype=float)
    var = float(np.var(vals))
    return 1.0 / (1.0 + var)

class IdentityState:
    def __init__(self):
        self.mean_intensity = 0.5
        self.mean_abs_error = 0.5
        self.mean_coherence = 0.5
        self.bias = 0.0
        self.baseline_a7 = 0.0
        self._initialized = False

    def update(self, intensity, error, coherence, a7_raw, alpha=0.05):
        ae = abs(error)
        if not self._initialized:
            self.mean_intensity = intensity
            self.mean_abs_error = ae
            self.mean_coherence = coherence
            self.bias = np.tanh(error)
            self.baseline_a7 = a7_raw
            self._initialized = True
            return

        self.mean_intensity = (1-alpha)*self.mean_intensity + alpha*intensity
        self.mean_abs_error = (1-alpha)*self.mean_abs_error + alpha*ae
        self.mean_coherence = (1-alpha)*self.mean_coherence + alpha*coherence
        self.bias = (1-alpha)*self.bias + alpha*np.tanh(error)
        self.baseline_a7 = (1-alpha)*self.baseline_a7 + alpha*a7_raw

    @property
    def vector(self):
        return np.array([
            self.mean_intensity,
            self.mean_abs_error,
            self.mean_coherence,
            self.bias
        ], dtype=float)

    def compute_lambda(self):
        # how much to pull towards baseline vs raw;
        # more bias or higher intensity → more regularization
        lam = 0.4 * abs(self.bias) + 0.3 * self.mean_intensity
        lam = max(0.0, min(0.85, lam))
        return lam

class C7BrainV9:
    def __init__(self):
        self.w1 = rng.normal(scale=0.1, size=(9,))
        self.w3 = rng.normal(scale=0.1, size=(9,))
        self.w5 = rng.normal(scale=0.1, size=(9,))
        self.b1 = 0.0
        self.b3 = 0.0
        self.b5 = 0.0
        self.identity = IdentityState()
        self.lr = 0.001

    def forward(self, audio_raw, text_raw, image_raw):
        a = audio_frontend(audio_raw)
        t = text_frontend(text_raw)
        i = image_frontend(image_raw)

        emb = np.concatenate([a, t, i], axis=0)

        energies = compute_energies(a, t, i)
        intensity = compute_intensity(energies)

        a1 = float(np.dot(self.w1, emb) + self.b1)
        a3 = float(np.dot(self.w3, emb) + self.b3)
        a5 = float(np.dot(self.w5, emb) + self.b5)

        a7_raw = (a1 + a3 + a5) / 3.0
        coherence = coherence_from_arrays(a1, a3, a5)

        if not self.identity._initialized:
            a7_final = a7_raw
        else:
            lam = self.identity.compute_lambda()
            a7_final = (1.0 - lam) * a7_raw + lam * self.identity.baseline_a7

        return {
            "a": a,
            "t": t,
            "i": i,
            "emb": emb,
            "energies": energies,
            "intensity": intensity,
            "a1": a1,
            "a3": a3,
            "a5": a5,
            "a7_raw": a7_raw,
            "a7_final": a7_final,
            "coherence": coherence,
        }

    def backward_step(self, emb, target, a1, a3, a5, a7_raw):
        # simple squared loss on raw output
        err = a7_raw - target
        dL_dA = 2.0 * err / 3.0

        grad_w1 = dL_dA * emb
        grad_w3 = dL_dA * emb
        grad_w5 = dL_dA * emb
        grad_b1 = dL_dA
        grad_b3 = dL_dA
        grad_b5 = dL_dA

        self.w1 -= self.lr * grad_w1
        self.w3 -= self.lr * grad_w3
        self.w5 -= self.lr * grad_w5
        self.b1 -= self.lr * grad_b1
        self.b3 -= self.lr * grad_b3
        self.b5 -= self.lr * grad_b5

        return float(err)

def sample_inputs():
    vals = [-1, 0, 1, 2, 3]
    a = rng.choice(vals, size=3)
    t = rng.choice(vals, size=3)
    i = rng.choice(vals, size=3)
    return a, t, i

def load_or_create_brain():
    if os.path.exists(BRAIN_PATH):
        with open(BRAIN_PATH, "rb") as f:
            brain = pickle.load(f)
        print("Loaded existing C7BrainV9 from file.")
    else:
        brain = C7BrainV9()
        print("No previous brain found. Creating NEW C7BrainV9.")
    return brain

def save_brain(brain):
    with open(BRAIN_PATH, "wb") as f:
        pickle.dump(brain, f)

def main():
    brain = load_or_create_brain()

    steps = 300
    print(f"Starting Phase 9 training for {steps} steps...\n")

    avg_loss = 0.0
    for step in range(1, steps + 1):
        a_raw, t_raw, i_raw = sample_inputs()
        out = brain.forward(a_raw, t_raw, i_raw)

        emb = out["emb"]
        a1, a3, a5 = out["a1"], out["a3"], out["a5"]
        a7_raw = out["a7_raw"]
        intensity = out["intensity"]
        coherence = out["coherence"]

        target = float(np.sum(emb))

        err = brain.backward_step(emb, target, a1, a3, a5, a7_raw)
        avg_loss = 0.9 * avg_loss + 0.1 * (err * err)

        brain.identity.update(intensity, err, coherence, a7_raw, alpha=0.05)

        if step % 50 == 0 or step == 1:
            print(
                f"Step {step:3d}/{steps} - "
                f"loss:{avg_loss:.4f} "
                f"int:{intensity:.3f} "
                f"coh:{coherence:.3f} "
                f"err:{err:.3f} "
                f"λ:{brain.identity.compute_lambda():.3f}"
            )

    print("\nPhase 9 training finished.\n")
    save_brain(brain)

    test_a = np.array([2, 1, 2])
    test_t = np.array([3, 3, 3])
    test_i = np.array([-1, 0, 1])

    out = brain.forward(test_a, test_t, test_i)
    emb = out["emb"]
    target = float(np.sum(emb))
    a7_raw = out["a7_raw"]
    a7_final = out["a7_final"]
    intensity = out["intensity"]
    coherence = out["coherence"]

    print("========== TEST AFTER PHASE 9 ==========\n")
    print("=== Raw Inputs ===")
    print(f"Audio Input : {list(test_a)}")
    print(f"Text Input  : {list(test_t)}")
    print(f"Image Input : {list(test_i)}\n")

    print("=== Frontend Outputs ===")
    print(f"AudioFrontend Output : {np.round(out['a'],3)}")
    print(f"TextFrontend Output  : {np.round(out['t'],3)}")
    print(f"ImageFrontend Output : {np.round(out['i'],3)}\n")

    print("=== Emb-C & Intensity ===")
    print(f"Emb-C Collapsed Vector: {np.round(emb,3)}")
    print(f"Target(sum Emb-C):      {target:.4f}")
    print(f"Intensity              : {intensity:.3f}")
    print(f"Energies               : { {k:round(v,3) for k,v in out['energies'].items()} }")
    print(f"Coherence              : {coherence:.3f}\n")

    print("=== Arrays & A7 ===")
    print(f"A1       : {out['a1']:.3f}")
    print(f"A3       : {out['a3']:.3f}")
    print(f"A5       : {out['a5']:.3f}")
    print(f"A7_raw   : {a7_raw:.3f}")
    print(f"A7_final : {a7_final:.3f}")
    print(f"Error_raw  : {a7_raw - target:.3f}")
    print(f"Error_final: {a7_final - target:.3f}\n")

    print("=== Identity (emergent profile) ===")
    print(f"mean_intensity : {brain.identity.mean_intensity:.3f}")
    print(f"mean_abs_error : {brain.identity.mean_abs_error:.3f}")
    print(f"mean_coherence : {brain.identity.mean_coherence:.3f}")
    print(f"bias           : {brain.identity.bias:.3f}")
    print(f"baseline_a7    : {brain.identity.baseline_a7:.3f}")
    print(f"identity_vec   : {np.round(brain.identity.vector,3)}")
    print(f"λ (current)    : {brain.identity.compute_lambda():.3f}")
    print("\n=====================================")

if __name__ == "__main__":
    main()