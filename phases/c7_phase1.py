# ==========================================================
#  C7-ASM — Prototype 1 (Full Minimal Version)
#  Multisensory → Emb-C → Prism → C7 Arrays → A7 Integrator
# ==========================================================

import random

# ==========================================================
# Debug helper
# ==========================================================
def debug(title, content):
    print(f"\n=== {title} ===")
    print(content)


# ----------------------------------------------------------
#  FRONTEND MODULES (3 senses)
# ----------------------------------------------------------
class AudioFrontend:
    def process(self, x):
        # Simple sensory-specific transform (avg-based)
        avg = sum(x) / len(x)
        out = [avg, avg * 0.5, avg * 1.5]
        debug("AudioFrontend Output", out)
        return out


class TextFrontend:
    def process(self, x):
        # Simple text transform (sum-based)
        s = sum(x)
        out = [s, s + 1, s - 1]
        debug("TextFrontend Output", out)
        return out


class ImageFrontend:
    def process(self, x):
        # Simple image transform (avg-max-min)
        avg = sum(x) / len(x)
        out = [avg, max(x), min(x)]
        debug("ImageFrontend Output", out)
        return out


# ----------------------------------------------------------
#  EmbeddingC — collapse multi-sensory vectors into 1 vector
# ----------------------------------------------------------
class EmbeddingC:
    def collapse(self, sensory_vectors):
        collapsed = []
        for vec in sensory_vectors:
            collapsed += vec
        debug("Emb-C Collapsed Vector", collapsed)
        return collapsed


# ----------------------------------------------------------
#  Prism-7 — route to a subset of C7 arrays
# ----------------------------------------------------------
class Prism7:
    def route(self, emb_c):
        debug("Prism-7 Input", emb_c)

        active = []
        for i in range(min(3, len(emb_c))):
            if emb_c[i] > 0:
                active.append((i * 2) + 1)   # routes to arrays: 1, 3, 5

        debug("Prism-7 Active Arrays", active)
        return active


# ----------------------------------------------------------
#  C7 Arrays — A1 ... A7
# ----------------------------------------------------------
class ArrayUnit:
    def __init__(self, id):
        self.id = id

    def forward(self, x):
        out = sum(x) + (self.id * 0.1)   # light unique behavior
        debug(f"Array A{self.id} Output", out)
        return out


# ----------------------------------------------------------
#  A7 Integrator — merges outputs
# ----------------------------------------------------------
class A7Integrator:
    def integrate(self, outputs):
        final = sum(outputs)
        debug("A7 Integrated Output", final)
        return final


# ==========================================================
#  RUN THE FULL PHASE 1 PROTOTYPE
# ==========================================================
if __name__ == "__main__":

    print("\n========== C7-ASM Prototype 1 — FULL VERSION ==========\n")

    # Fake sensory inputs
    audio_input = [random.randint(-1, 3) for _ in range(3)]
    text_input  = [random.randint(-1, 3) for _ in range(3)]
    image_input = [random.randint(-1, 3) for _ in range(3)]

    debug("Audio Input", audio_input)
    debug("Text Input",  text_input)
    debug("Image Input", image_input)

    # Stage 1 — Sensory frontends
    a_vec = AudioFrontend().process(audio_input)
    t_vec = TextFrontend().process(text_input)
    i_vec = ImageFrontend().process(image_input)

    # Stage 2 — EmbeddingC
    emb_c = EmbeddingC().collapse([a_vec, t_vec, i_vec])

    # Stage 3 — Prism-7 routing
    active_arrays = Prism7().route(emb_c)

    # Stage 4 — Run selected C7 arrays
    outputs = []
    for arr_id in active_arrays:
        arr = ArrayUnit(arr_id)
        outputs.append(arr.forward(emb_c))

    # Stage 5 — A7 integration
    final_output = A7Integrator().integrate(outputs)

    print("\n========== FINAL OUTPUT ==========")
    print(final_output)
    print("=====================================\n")