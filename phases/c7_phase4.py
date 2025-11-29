# ==========================================================
#  C7-ASM — Prototype 4 (Dynamic Prism + Learning Arrays)
#  - Multisensory frontends (Audio/Text/Image)
#  - EmbeddingC (Emb-C)
#  - Dynamic Prism-7 routing based on modality energy
#  - Learnable C7 Arrays (مثل فاز ۳)
#  - Target: sum(Emb-C)  (فعلاً ساده، بعداً عوضش می‌کنیم)
# ==========================================================

import random

VERBOSE = False


def debug(title, content):
    if VERBOSE:
        print(f"\n=== {title} ===")
        print(content)


# -------- Frontends --------
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


# -------- Dynamic Prism-7 (Phase 4) --------
class DynamicPrism7:
    """
    - محاسبه انرژی هر حس
    - انتخاب modality غالب
    - انتخاب تعداد ستون‌ها بر اساس شدت غالب
    - نگاشت modality → ستون‌های ترجیحی
    """

    def route(self, emb_c, a_vec, t_vec, i_vec):
        debug("Prism-7 Input (Emb-C)", emb_c)

        audio_energy = sum(abs(x) for x in a_vec)
        text_energy = sum(abs(x) for x in t_vec)
        image_energy = sum(abs(x) for x in i_vec)
        total = audio_energy + text_energy + image_energy

        if total <= 1e-8:
            # fallback
            dominant = "text"
            intensity = 0.0
        else:
            energies = {
                "audio": audio_energy,
                "text": text_energy,
                "image": image_energy,
            }
            dominant = max(energies, key=energies.get)
            intensity = energies[dominant] / total

        debug(
            "Modality Energies",
            {
                "audio": audio_energy,
                "text": text_energy,
                "image": image_energy,
                "dominant": dominant,
                "intensity": intensity,
            },
        )

        # نگاشت modality غالب → ستون‌های ترجیحی
        if dominant == "text":
            preferred = [1, 3, 5]
        elif dominant == "audio":
            preferred = [2, 4, 6]
        else:  # image
            preferred = [3, 5, 7]

        # تعداد ستون‌های فعال بر اساس شدت
        if intensity > 0.65:
            k = 3
        elif intensity > 0.4:
            k = 2
        else:
            k = 1

        active = preferred[:k]
        debug("Prism-7 Active Arrays (dynamic)", active)
        return active, dominant, intensity


# -------- Learnable Array Unit --------
class ArrayUnit:
    def __init__(self, id, dim):
        self.id = id
        # وزن‌های اولیه رندوم کوچک
        self.w = [random.uniform(-0.1, 0.1) for _ in range(dim)]
        self.b = 0.0

    def forward(self, x):
        s = 0.0
        for wi, xi in zip(self.w, x):
            s += wi * xi
        out = s + self.b
        debug(f"Array A{self.id} Output", out)
        return out

    def update(self, x, target, pred, lr):
        """
        گرادیان ساده:
        loss = 0.5 * (pred - target)^2
        dL/dw = (pred - target) * x
        dL/db = (pred - target)
        """
        error = pred - target
        for i in range(len(self.w)):
            self.w[i] -= lr * error * x[i]
        self.b -= lr * error
        return 0.5 * (error ** 2)


# -------- A7 Integrator --------
class A7Integrator:
    def integrate(self, outputs):
        if not outputs:
            final = 0.0
        else:
            final = sum(outputs) / len(outputs)
        debug("A7 Integrated Output", final)
        return final


# -------- C7 Model (Phase 4) --------
class C7Model:
    def __init__(self, emb_dim=9):
        self.audio_frontend = AudioFrontend()
        self.text_frontend = TextFrontend()
        self.image_frontend = ImageFrontend()
        self.emb_c_module = EmbeddingC()
        self.prism = DynamicPrism7()
        self.a7 = A7Integrator()
        self.arrays = {i: ArrayUnit(i, emb_dim) for i in range(1, 8)}

    def forward(self, audio_input, text_input, image_input):
        # 1) حس‌ها
        a_vec = self.audio_frontend.process(audio_input)
        t_vec = self.text_frontend.process(text_input)
        i_vec = self.image_frontend.process(image_input)

        # 2) Emb-C
        emb_c = self.emb_c_module.collapse([a_vec, t_vec, i_vec])

        # 3) Prism داینامیک
        active_ids, dominant, intensity = self.prism.route(
            emb_c, a_vec, t_vec, i_vec
        )

        # 4) آرایه‌های فعال
        outputs = {}
        for arr_id in active_ids:
            arr = self.arrays[arr_id]
            outputs[arr_id] = arr.forward(emb_c)

        # 5) ادغام در A7
        final_output = self.a7.integrate(list(outputs.values()))

        return emb_c, active_ids, outputs, final_output, dominant, intensity, a_vec, t_vec, i_vec

    def train(self, steps=300, lr=0.01):
        global VERBOSE
        VERBOSE = False
        print(f"Starting Phase 4 training for {steps} steps...")

        recent_losses = []

        for step in range(steps):
            audio_input = [random.randint(-1, 3) for _ in range(3)]
            text_input = [random.randint(-1, 3) for _ in range(3)]
            image_input = [random.randint(-1, 3) for _ in range(3)]

            emb_c, active_ids, outputs, final_output, dom, inten, a_vec, t_vec, i_vec = \
                self.forward(audio_input, text_input, image_input)

            if not active_ids:
                continue

            # تارگت ساده
            target = sum(emb_c)

            total_loss = 0.0
            for arr_id in active_ids:
                arr = self.arrays[arr_id]
                pred = outputs[arr_id]
                loss = arr.update(emb_c, target, pred, lr)
                total_loss += loss

            recent_losses.append(total_loss)

            if (step + 1) % 50 == 0 and recent_losses:
                avg_loss = sum(recent_losses[-50:]) / len(recent_losses[-50:])
                print(f"Step {step+1}/{steps} - avg loss: {avg_loss:.4f}")

        print("Phase 4 training finished.\n")


# ==========================================================
#  RUN PHASE 4
# ==========================================================
if __name__ == "__main__":

    model = C7Model(emb_dim=9)
    model.train(steps=300, lr=0.01)

    VERBOSE = True
    print("\n========== TEST AFTER PHASE 4 TRAINING ==========\n")

    audio_input = [random.randint(-1, 3) for _ in range(3)]
    text_input = [random.randint(-1, 3) for _ in range(3)]
    image_input = [random.randint(-1, 3) for _ in range(3)]

    debug("Audio Input", audio_input)
    debug("Text Input", text_input)
    debug("Image Input", image_input)

    emb_c, active_ids, outputs, final_output, dom, inten, a_vec, t_vec, i_vec = \
        model.forward(audio_input, text_input, image_input)

    target = sum(emb_c)
    print(f"\nDominant modality : {dom}")
    print(f"Intensity         : {inten:.3f}")
    print(f"Active arrays     : {active_ids}")
    print(f"\nTarget (sum(Emb-C)): {target:.4f}")
    print(f"Final Output (A7):   {final_output:.4f}")

    print("\nPer-array predictions:")
    for arr_id in sorted(outputs.keys()):
        print(f"  A{arr_id}: {outputs[arr_id]:.4f}")

    error = final_output - target
    print(f"\nFinal error (A7 - target): {error:.4f}")
    print("\n=========================================\n")