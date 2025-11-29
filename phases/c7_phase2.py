# ==========================================================
#  C7-ASM — Prototype 2 (Visualization Enabled, Clean Indent)
# ==========================================================

import random

# -------- Debug Print --------
def debug(title, content):
    print(f"\n=== {title} ===")
    print(content)


# -------- Frontend Modules --------
class AudioFrontend:
    def process(self, x):
        avg = sum(x) / len(x)
        out = [avg, avg * 0.5, avg * 1.5]
        debug("AudioFrontend Output", out)
        return out


class TextFrontend:
    def process(self, x):
        s = sum(x)
        out = [s, s + 1, s - 1]
        debug("TextFrontend Output", out)
        return out


class ImageFrontend:
    def process(self, x):
        avg = sum(x) / len(x)
        out = [avg, max(x), min(x)]
        debug("ImageFrontend Output", out)
        return out


# -------- EmbeddingC --------
class EmbeddingC:
    def collapse(self, sensory_vectors):
        collapsed = []
        for vec in sensory_vectors:
            collapsed += vec
        debug("Emb-C Collapsed Vector", collapsed)
        return collapsed


# -------- Prism-7 --------
class Prism7:
    def route(self, emb_c):
        debug("Prism-7 Input", emb_c)
        active = []
        for i in range(min(3, len(emb_c))):
            if emb_c[i] > 0:
                active.append((i * 2) + 1)
        debug("Prism-7 Active Arrays", active)
        return active


# -------- Array Units --------
class ArrayUnit:
    def __init__(self, id):
        self.id = id

    def forward(self, x):
        out = sum(x) + (self.id * 0.1)
        debug(f"Array A{self.id} Output", out)
        return out


# -------- A7 Integrator --------
class A7Integrator:
    def integrate(self, outputs):
        final = sum(outputs)
        debug("A7 Integrated Output", final)
        return final


# -------- Visualization: Array Activations --------
def visualize_array_activations(array_outputs):
    print("\n=== Array Activations (Normalized) ===")

    if not array_outputs:
        print("No active arrays.")
        return

    vals = list(array_outputs.values())
    mn = min(vals)
    mx = max(vals)

    normalized = {}
    if mx == mn:
        for k in array_outputs:
            normalized[k] = 1.0
    else:
        for k, v in array_outputs.items():
            normalized[k] = (v - mn) / (mx - mn)

    for i in range(1, 7 + 1):
        if i in array_outputs:
            n = normalized[i]
            bar = "█" * max(1, int(n * 10))
            print(f"A{i}: {bar} ({array_outputs[i]:.3f})")
        else:
            print(f"A{i}: (inactive)")


# -------- Visualization: Modality Contribution --------
def visualize_modality_contribution(a_vec, t_vec, i_vec):
    ae = sum(abs(x) for x in a_vec)
    te = sum(abs(x) for x in t_vec)
    ie = sum(abs(x) for x in i_vec)

    total = ae + te + ie
    if total == 0:
        ap = tp = ip = 0
    else:
        ap = (ae / total) * 100
        tp = (te / total) * 100
        ip = (ie / total) * 100

    print("\n=== Modality Contribution (%) ===")
    print(f"Audio : {ap:5.1f}%")
    print(f"Text  : {tp:5.1f}%")
    print(f"Image : {ip:5.1f}%")


# -------- Visualization: Brain Map --------
def visualize_brain_map(active_arrays):
    print("\n=== Brain Map (Active Arrays) ===")
    for i in range(1, 8):
        if i in active_arrays:
            print(f"A{i}: ███ ACTIVE")
        else:
            print(f"A{i}: ...")


# ==========================================================
#  RUN PROTOTYPE
# ==========================================================
if __name__ == "__main__":

    print("\n========== C7-ASM Prototype 2 — VISUAL CONSOLE ==========\n")

    # Fake sensory inputs
    audio_input = [random.randint(-1, 3) for _ in range(3)]
    text_input  = [random.randint(-1, 3) for _ in range(3)]
    image_input = [random.randint(-1, 3) for _ in range(3)]

    debug("Audio Input", audio_input)
    debug("Text Input", text_input)
    debug("Image Input", image_input)

    # Stage 1 – frontends
    a_vec = AudioFrontend().process(audio_input)
    t_vec = TextFrontend().process(text_input)
    i_vec = ImageFrontend().process(image_input)

    # Phase 2 visualization
    visualize_modality_contribution(a_vec, t_vec, i_vec)

    # Stage 2 – EmbeddingC
    emb_c = EmbeddingC().collapse([a_vec, t_vec, i_vec])

    # Stage 3 – Prism routing
    active_arrays = Prism7().route(emb_c)

    # Brain map (Phase 2)
    visualize_brain_map(active_arrays)

    # Stage 4 – Arrays
    array_outputs = {}
    for arr_id in active_arrays:
        arr = ArrayUnit(arr_id)
        array_outputs[arr_id] = arr.forward(emb_c)

    # Array activation visualization
    visualize_array_activations(array_outputs)

    # Stage 5 – A7
    final_output = A7Integrator().integrate(list(array_outputs.values()))

    print("\n========== FINAL OUTPUT ==========")
    print(final_output)
    print("=====================================\n")