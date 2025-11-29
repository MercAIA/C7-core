import numpy as np
import json
import os

# ============================================================
#  Phase 7.4 – Architecture C + 3-Mode Operational System
# ============================================================

class C7BrainV74:
    def __init__(self):
        self.W1 = np.random.randn(9, 16) * 0.1
        self.W3 = np.random.randn(9, 16) * 0.1
        self.W5 = np.random.randn(9, 16) * 0.1
        self.WF = np.random.randn(16, 1) * 0.1

        self.mem_A7 = []
        self.mem_intensity = []

        self.prev_mode = "A"

    def save(self, path="c7_brain_v74.json"):
        data = {
            "W1": self.W1.tolist(),
            "W3": self.W3.tolist(),
            "W5": self.W5.tolist(),
            "WF": self.WF.tolist()
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load(path="c7_brain_v74.json"):
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        brain = C7BrainV74()
        brain.W1 = np.array(data["W1"])
        brain.W3 = np.array(data["W3"])
        brain.W5 = np.array(data["W5"])
        brain.WF = np.array(data["WF"])
        return brain

    # -------------------------------
    #  3 Frontends: Audio / Text / Image
    # -------------------------------
    def audio_front(self, a):
        return np.array([a[0]+1, a[1]+0.5, a[2]+1.5])

    def text_front(self, t):
        return np.array([t[0]*3, t[1]*3, t[2]*3+1])

    def image_front(self, i):
        return np.array([i[0]*0.5, i[1]+1, i[2]*0.5])

    # -------------------------------
    # Mode selector
    # -------------------------------
    def select_mode(self, intensity, coherence, error):
        if intensity < 0.45 and coherence > 0.75:
            return "A"       # stable
        if intensity > 0.75 or error > 3:
            return "C"       # deep
        return "B"           # dynamic

    # -------------------------------
    # Mode parameters
    # -------------------------------
    def get_mode_params(self, mode):
        if mode == "A":
            return dict(gain=0.90, mem_depth=2, smooth=0.06)
        if mode == "B":
            return dict(gain=1.00, mem_depth=5, smooth=0.12)
        if mode == "C":
            return dict(gain=1.15, mem_depth=9, smooth=0.25)

    # -------------------------------
    # Horizontal coherence
    # -------------------------------
    def coherence(self, a1, a3, a5):
        v = np.var([a1, a3, a5])
        return 1 / (1 + v)

    # -------------------------------
    # Forward pass
    # -------------------------------
    def forward(self, audio, text, image):

        A = self.audio_front(audio)
        T = self.text_front(text)
        I = self.image_front(image)

        emb = np.concatenate([A, T, I])

        energies = {
            "audio": np.linalg.norm(A),
            "text": np.linalg.norm(T),
            "image": np.linalg.norm(I)
        }

        intensity = np.linalg.norm(emb) / 40
        dom = max(energies, key=energies.get)

        a1 = float(emb @ self.W1 @ np.ones(16))
        a3 = float(emb @ self.W3 @ np.ones(16))
        a5 = float(emb @ self.W5 @ np.ones(16))

        a7_raw = float(np.mean([a1, a3, a5]))

        coh = self.coherence(a1, a3, a5)

        target = np.sum(emb)
        error = a7_raw - target

        mode = self.select_mode(intensity, coh, abs(error))
        params = self.get_mode_params(mode)

        gain = params["gain"]
        mem_depth = params["mem_depth"]
        smooth = params["smooth"]

        # Scaling
        a1 *= gain
        a3 *= gain
        a5 *= gain

        a7_scaled = np.mean([a1, a3, a5])

        # Memory
        self.mem_A7.append(a7_scaled)
        self.mem_intensity.append(intensity)
        if len(self.mem_A7) > 30:
            self.mem_A7.pop(0)
            self.mem_intensity.pop(0)

        mem_avg = (
            smooth * np.mean(self.mem_A7[-mem_depth:])
            + (1 - smooth) * a7_scaled
        )

        final = mem_avg

        return {
            "A": A.tolist(),
            "T": T.tolist(),
            "I": I.tolist(),
            "emb": emb.tolist(),
            "intensity": float(intensity),
            "energies": energies,
            "mode": mode,
            "coherence": float(coh),
            "a1": float(a1),
            "a3": float(a3),
            "a5": float(a5),
            "a7_raw": float(a7_raw),
            "a7_final": float(final),
            "target": float(target),
            "error": float(final - target),
            "params": params
        }


# ============================================================
# Training Loop (simple)
# ============================================================

def train_phase7_4():
    brain = C7BrainV74.load()
    if brain is None:
        print("No previous brain found. Creating NEW C7BrainV7.4.")
        brain = C7BrainV74()

    print("Starting training…")

    for step in range(300):
        audio = np.random.randint(-1, 3, 3)
        text  = np.random.randint(-1, 4, 3)
        image = np.random.randint(-1, 3, 3)

        out = brain.forward(audio, text, image)

        loss = abs(out["error"])
        if (step+1) % 50 == 0:
            print(f"Step {step+1}/300 – loss: {loss:.4f} mode:{out['mode']} coh:{out['coherence']:.2f}")

    brain.save()
    print("\nTraining finished. Brain saved.\n")
    return brain


# ============================================================
# Run once & test
# ============================================================

if __name__ == "__main__":
    brain = train_phase7_4()

    audio = [2,1,2]
    text  = [3,3,3]
    image = [-1,0,1]

    out = brain.forward(audio, text, image)

    print("========== TEST OUTPUT (Phase 7.4) ==========")
    for k,v in out.items():
        print(k,":",v)