import random
import json
import os

# ---------- Frontend: simple multi-modal transforms ----------

class MultiModalFrontend:
    def audio_frontend(self, x):
        return [v + 1.0 for v in x]

    def text_frontend(self, x):
        return [(v + 1.0) * 3.0 for v in x]

    def image_frontend(self, x):
        return [(v + 1.0) * 2.0 for v in x]


# ---------- Embedding Collapser ----------

class EmbeddingCollapser:
    def collapse(self, audio_vec, text_vec, image_vec):
        return audio_vec + text_vec + image_vec


# ---------- Dynamic Prism-7 (routing + intensity) ----------

class DynamicPrism7:
    def __init__(self):
        self.modality_map = {
            "audio": [2, 4, 6],
            "text":  [1, 3, 5],
            "image": [3, 5, 7],
        }

    def route(self, audio_vec, text_vec, image_vec):
        audio_energy = sum(abs(v) for v in audio_vec)
        text_energy  = sum(abs(v) for v in text_vec)
        image_energy = sum(abs(v) for v in image_vec)

        energies = {
            "audio": audio_energy,
            "text":  text_energy,
            "image": image_energy,
        }

        total = audio_energy + text_energy + image_energy
        if total <= 1e-9:
            dominant = "text"
            intensity = 0.0
        else:
            dominant = max(energies, key=energies.get)
            intensity = energies[dominant] / total

        base_indices = self.modality_map[dominant]
        if intensity <= 0.40:
            active = [base_indices[1]]
        elif intensity <= 0.65:
            active = base_indices[:2]
        else:
            active = base_indices

        return {
            "dominant": dominant,
            "intensity": intensity,
            "energies": energies,
            "active_arrays": active,
        }


# ---------- Array Unit ----------

class ArrayUnit:
    def __init__(self, input_dim, weights=None, bias=None):
        if weights is not None and bias is not None:
            self.weights = list(weights)
            self.bias = float(bias)
        else:
            self.weights = [random.uniform(-0.5, 0.5) for _ in range(input_dim)]
            self.bias = random.uniform(-0.1, 0.1)

    def forward(self, x):
        return sum(w * v for w, v in zip(self.weights, x)) + self.bias

    def update(self, x, target, pred, lr):
        grad = (pred - target)
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grad * x[i]
        self.bias -= lr * grad


# ---------- Smart Memory + Soft Sync ----------

class SmartMemoryController:
    def __init__(self, history_len=16, sync_eps=0.8, sync_lambda=0.15):
        self.history_len = history_len
        self.sync_eps = sync_eps
        self.sync_lambda = sync_lambda
        self.history = []

    def update_history(self, a7_out, target, intensity, preds):
        self.history.append((a7_out, target, intensity, list(preds)))
        if len(self.history) > self.history_len:
            self.history.pop(0)

    def memory_weight(self, intensity):
        return max(0.0, min(1.0, 1.0 - intensity))

    def soft_sync(self, preds, intensity):
        if not preds:
            return preds

        m = sum(preds) / len(preds)
        mw = self.memory_weight(intensity)
        if mw <= 1e-6:
            return preds

        synced = []
        for p in preds:
            delta = p - m
            if abs(delta) > self.sync_eps:
                p_adj = p - self.sync_lambda * mw * delta
                synced.append(p_adj)
            else:
                synced.append(p)
        return synced


# ---------- Save / Load Arrays (persistent brain) ----------

def save_arrays(arrays, path="c7_phase5_weights.json"):
    data = {}
    for idx, arr in arrays.items():
        data[str(idx)] = {
            "weights": arr.weights,
            "bias": arr.bias,
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_arrays(input_dim, path="c7_phase5_weights.json"):
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    arrays = {}
    for key, val in data.items():
        idx = int(key)
        arrays[idx] = ArrayUnit(
            input_dim,
            weights=val["weights"],
            bias=val["bias"]
        )
    # اگر وزن‌ها ناقص بود، بقیه را رندوم بساز
    for idx in range(1, 8):
        if idx not in arrays:
            arrays[idx] = ArrayUnit(input_dim)
    return arrays


# ---------- Training Phase 5 (no capacity scaling) ----------

def train_phase5(steps=300, lr=0.001):
    frontend = MultiModalFrontend()
    collapser = EmbeddingCollapser()
    prism = DynamicPrism7()
    memory = SmartMemoryController()

    input_dim = 9
    arrays = load_arrays(input_dim)
    if arrays is None:
        arrays = {i: ArrayUnit(input_dim) for i in range(1, 8)}
        print("No previous brain found. Creating a NEW brain.\n")
    else:
        print("Loaded existing brain from file.\n")

    print(f"Starting Phase 5 training for {steps} steps...")
    avg_loss = 0.0

    for step in range(1, steps + 1):
        audio_in = [random.randint(-1, 3) for _ in range(3)]
        text_in  = [random.randint(-1, 3) for _ in range(3)]
        image_in = [random.randint(-1, 3) for _ in range(3)]

        a_vec = frontend.audio_frontend(audio_in)
        t_vec = frontend.text_frontend(text_in)
        i_vec = frontend.image_frontend(image_in)

        emb_c = collapser.collapse(a_vec, t_vec, i_vec)
        routing = prism.route(a_vec, t_vec, i_vec)
        dominant = routing["dominant"]
        intensity = routing["intensity"]
        active_ids = routing["active_arrays"]

        eff_lr = lr  # حوضچه ظرفیت فعلاً خاموش

        preds_raw = [arrays[idx].forward(emb_c) for idx in active_ids]
        a7_raw = sum(preds_raw) / len(preds_raw)

        target = sum(emb_c)

        memory.update_history(a7_raw, target, intensity, preds_raw)
        preds_synced = memory.soft_sync(preds_raw, intensity)
        a7_synced = sum(preds_synced) / len(preds_synced)

        loss = 0.0
        for p in preds_synced:
            loss += 0.5 * (p - target) ** 2
        loss /= len(preds_synced)
        avg_loss = 0.9 * avg_loss + 0.1 * loss if step > 1 else loss

        for idx, p_sync in zip(active_ids, preds_synced):
            arrays[idx].update(emb_c, target, p_sync, eff_lr)

        if step % 50 == 0 or step == steps:
            print(f"Step {step}/{steps} - avg loss: {avg_loss:.4f}")

    print("Phase 5 training finished.\n")

    # ذخیره مغز
    save_arrays(arrays)
    print("Brain weights saved.\n")

    return frontend, collapser, prism, memory, arrays


# ---------- Test Phase 5 ----------

def test_phase5(frontend, collapser, prism, memory, arrays):
    print("\n========== TEST AFTER PHASE 5 TRAINING ==========\n")

    audio_in = [2, 1, 2]
    text_in  = [3, 3, 3]
    image_in = [-1, 0, 1]

    print("=== Raw Inputs ===")
    print("Audio Input :", audio_in)
    print("Text Input  :", text_in)
    print("Image Input :", image_in)

    a_vec = frontend.audio_frontend(audio_in)
    t_vec = frontend.text_frontend(text_in)
    i_vec = frontend.image_frontend(image_in)

    print("\n=== Frontend Outputs ===")
    print("AudioFrontend Output :", a_vec)
    print("TextFrontend Output  :", t_vec)
    print("ImageFrontend Output :", i_vec)

    emb_c = collapser.collapse(a_vec, t_vec, i_vec)
    print("\n=== Emb-C Collapsed Vector ===")
    print(emb_c)

    routing = prism.route(a_vec, t_vec, i_vec)
    dominant = routing["dominant"]
    intensity = routing["intensity"]
    energies = routing["energies"]
    active_ids = routing["active_arrays"]

    print("\n=== Prism-7 Routing ===")
    print("Energies       :", energies)
    print("Dominant       :", dominant)
    print("Intensity      :", f"{intensity:.3f}")
    print("Active arrays  :", active_ids)

    preds_raw = [arrays[idx].forward(emb_c) for idx in active_ids]
    a7_raw = sum(preds_raw) / len(preds_raw)

    preds_synced = memory.soft_sync(preds_raw, intensity)
    a7_synced = sum(preds_synced) / len(preds_synced)

    target = sum(emb_c)

    print("\n=== Array Outputs (RAW) ===")
    for idx, p in zip(active_ids, preds_raw):
        print(f"A{idx}: {p:.4f}")
    print("A7 RAW (mean)  :", f"{a7_raw:.4f}")

    print("\n=== Array Outputs (SYNCED) ===")
    for idx, p in zip(active_ids, preds_synced):
        print(f"A{idx} (synced): {p:.4f}")
    print("A7 SYNCED      :", f"{a7_synced:.4f}")

    print("\n=== Target & Errors ===")
    print("Target (sum Emb-C):", f"{target:.4f}")
    print("Error RAW   (A7_raw - target):   ", f"{a7_raw - target:.4f}")
    print("Error SYNC  (A7_sync - target):  ", f"{a7_synced - target:.4f}")

    print("\n==========================================")


if __name__ == "__main__":
    frontend, collapser, prism, memory, arrays = train_phase5(steps=300, lr=0.001)
    test_phase5(frontend, collapser, prism, memory, arrays)