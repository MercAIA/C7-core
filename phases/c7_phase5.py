import random
import math

# ---------- Frontend: simple multi-modal transforms ----------

class MultiModalFrontend:
    def audio_frontend(self, x):
        # Shift to non-negative, light scaling
        return [v + 1.0 for v in x]

    def text_frontend(self, x):
        # Stronger scaling to dominate when large
        return [(v + 1.0) * 3.0 for v in x]

    def image_frontend(self, x):
        # Medium scaling
        return [(v + 1.0) * 2.0 for v in x]


# ---------- Embedding Collapser ----------

class EmbeddingCollapser:
    def collapse(self, audio_vec, text_vec, image_vec):
        # Simple concat
        return audio_vec + text_vec + image_vec


# ---------- Dynamic Prism-7 (routing + intensity) ----------

class DynamicPrism7:
    def __init__(self):
        # mapping: modality -> preferred array indices (1-based)
        self.modality_map = {
            "audio": [2, 4, 6],
            "text":  [1, 3, 5],
            "image": [3, 5, 7],
        }

    def route(self, audio_vec, text_vec, image_vec):
        # energies = L1 norm
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

        # choose active arrays based on intensity
        base_indices = self.modality_map[dominant]
        if intensity <= 0.40:
            active = [base_indices[1]]  # only middle
        elif intensity <= 0.65:
            active = base_indices[:2]   # two
        else:
            active = base_indices       # three

        return {
            "dominant": dominant,
            "intensity": intensity,
            "energies": energies,
            "active_arrays": active,
        }


# ---------- Array Unit (simple linear neuron) ----------

class ArrayUnit:
    def __init__(self, input_dim):
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(input_dim)]
        self.bias = random.uniform(-0.1, 0.1)

    def forward(self, x):
        return sum(w * v for w, v in zip(self.weights, x)) + self.bias

    def update(self, x, target, pred, lr):
        # d(loss)/d(pred) = (pred - target)
        grad = (pred - target)
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grad * x[i]
        self.bias -= lr * grad


# ---------- Smart Memory + Soft Sync Controller (Phase 5) ----------

class SmartMemoryController:
    def __init__(self, history_len=16, sync_eps=0.8, sync_lambda=0.15):
        self.history_len = history_len
        self.sync_eps = sync_eps      # آستانه سوپاپ
        self.sync_lambda = sync_lambda
        self.history = []  # store (a7_out, target, intensity, preds)

    def update_history(self, a7_out, target, intensity, preds):
        self.history.append((a7_out, target, intensity, list(preds)))
        if len(self.history) > self.history_len:
            self.history.pop(0)

    def memory_weight(self, intensity):
        # High intensity → low memory influence (رابطه عکس)
        return max(0.0, min(1.0, 1.0 - intensity))

    def soft_sync(self, preds, intensity):
        """
        سینک خیلی شل:
        - فقط اوت‌لایرها رو کمی به میانگین نزدیک می‌کنه
        - وقتی شدت بالاست، تأثیر حافظه کم می‌شه
        """
        if not preds:
            return preds

        m = sum(preds) / len(preds)
        mw = self.memory_weight(intensity)
        if mw <= 1e-6:
            return preds  # حافظه عملاً خاموش

        synced = []
        for p in preds:
            delta = p - m
            if abs(delta) > self.sync_eps:
                # Soft pull toward mean
                p_adj = p - self.sync_lambda * mw * delta
                synced.append(p_adj)
            else:
                synced.append(p)
        return synced


# ---------- C7-ASM Phase 5 Prototype ----------

def train_phase5(steps=300, lr=0.001):
    frontend = MultiModalFrontend()
    collapser = EmbeddingCollapser()
    prism = DynamicPrism7()
    memory = SmartMemoryController()

    input_dim = 9
    arrays = {i: ArrayUnit(input_dim) for i in range(1, 8)}

    print(f"Starting Phase 5 training for {steps} steps...")
    avg_loss = 0.0

    for step in range(1, steps + 1):
        # random toy inputs in range [-1, 3]
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

        # capacity scaling: شدت بالا → ظرفیت بیشتر → lr کمی بیشتر
        capacity_scale = 1.0   # فعلا حوضچه رو خاموش می‌کنیم
        eff_lr = lr

        # forward through active arrays
        preds_raw = []
        for idx in active_ids:
            preds_raw.append(arrays[idx].forward(emb_c))

        # A7 raw output
        a7_raw = sum(preds_raw) / len(preds_raw)

        # target: همچنان sum(Emb-C) به‌عنوان هدف sanity
        target = sum(emb_c)

        # update memory
        memory.update_history(a7_raw, target, intensity, preds_raw)

        # apply very loose sync (سوپاپ اطمینان)
        preds_synced = memory.soft_sync(preds_raw, intensity)
        a7_synced = sum(preds_synced) / len(preds_synced)

        # loss based on synced predictions
        loss = 0.0
        for p in preds_synced:
            loss += 0.5 * (p - target) ** 2
        loss /= len(preds_synced)
        avg_loss = 0.9 * avg_loss + 0.1 * loss if step > 1 else loss

        # update arrays with synced predictions
        for idx, p_sync in zip(active_ids, preds_synced):
            arrays[idx].update(emb_c, target, p_sync, eff_lr)

        if step % 50 == 0 or step == steps:
            print(f"Step {step}/{steps} - avg loss: {avg_loss:.4f}")

    print("Phase 5 training finished.\n")
    return frontend, collapser, prism, memory, arrays


def test_phase5(frontend, collapser, prism, memory, arrays):
    print("\n========== TEST AFTER PHASE 5 TRAINING ==========\n")

    # مثال: TEXT قوی، AUDIO متوسط، IMAGE ضعیف
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

    # raw predictions
    preds_raw = [arrays[idx].forward(emb_c) for idx in active_ids]
    a7_raw = sum(preds_raw) / len(preds_raw)

    # synced predictions
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