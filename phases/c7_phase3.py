# ==========================================================
#  C7-ASM — Prototype 3 (Learning Enabled)
#  - Multisensory frontends (Audio/Text/Image)
#  - EmbeddingC (Emb-C)
#  - Prism-7 routing
#  - C7 Arrays with learnable weights
#  - Simple training loop to match target = sum(Emb-C)
# ==========================================================

import random

# -------- Global verbose flag --------
VERBOSE = False


def debug(title, content):
    """Print debug info only if VERBOSE is True."""
    if VERBOSE:
        print(f"\n=== {title} ===")
        print(content)


# -------- Frontend Modules (3 senses) --------
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
        # قانون ساده: سه عنصر اول Emb-C را چک کن
        for i in range(min(3, len(emb_c))):
            if emb_c[i] > 0:
                active.append((i * 2) + 1)  # 1,3,5
        debug("Prism-7 Active Arrays", active)
        return active


# -------- Learnable Array Unit --------
class ArrayUnit:
    def __init__(self, id, dim):
        self.id = id
        # وزن‌های اولیه کوچک رندوم
        self.w = [random.uniform(-0.1, 0.1) for _ in range(dim)]
        self.b = 0.0

    def forward(self, x):
        # dot(w, x) + b
        s = 0.0
        for wi, xi in zip(self.w, x):
            s += wi * xi
        out = s + self.b
        debug(f"Array A{self.id} Output", out)
        return out

    def update(self, x, target, pred, lr):
        """
        آپدیت ساده با گرادیان:
        loss = 0.5 * (pred - target)^2
        dL/dw = (pred - target) * x
        dL/db = (pred - target)
        """
        error = pred - target
        # آپدیت وزن‌ها
        for i in range(len(self.w)):
            self.w[i] -= lr * error * x[i]
        # آپدیت بایاس
        self.b -= lr * error
        # برگرداندن loss
        return 0.5 * (error ** 2)


# -------- A7 Integrator --------
class A7Integrator:
    def integrate(self, outputs):
        # فعلاً میانگین خروجی آرایه‌های فعال
        if not outputs:
            final = 0.0
        else:
            final = sum(outputs) / len(outputs)
        debug("A7 Integrated Output", final)
        return final


# -------- C7 Model Wrapper --------
class C7Model:
    def __init__(self, emb_dim=9):
        self.audio_frontend = AudioFrontend()
        self.text_frontend = TextFrontend()
        self.image_frontend = ImageFrontend()
        self.emb_c_module = EmbeddingC()
        self.prism = Prism7()
        self.a7 = A7Integrator()

        # 7 آرایه با وزن‌های یادگیرنده
        self.arrays = {i: ArrayUnit(i, emb_dim) for i in range(1, 8)}

    def forward(self, audio_input, text_input, image_input):
        # frontends
        a_vec = self.audio_frontend.process(audio_input)
        t_vec = self.text_frontend.process(text_input)
        i_vec = self.image_frontend.process(image_input)

        # Emb-C
        emb_c = self.emb_c_module.collapse([a_vec, t_vec, i_vec])

        # Prism
        active_ids = self.prism.route(emb_c)

        # Arrays
        outputs = {}
        for arr_id in active_ids:
            arr = self.arrays[arr_id]
            outputs[arr_id] = arr.forward(emb_c)

        # A7 integration
        final_output = self.a7.integrate(list(outputs.values()))

        return emb_c, active_ids, outputs, final_output

    def train(self, steps=300, lr=0.01):
        global VERBOSE
        VERBOSE = False  # حین آموزش خروجی دیباگ نمی‌خواهیم

        print(f"Starting training for {steps} steps (Phase 3)...")
        recent_losses = []

        for step in range(steps):
            # ورودی‌های رندوم حسی
            audio_input = [random.randint(-1, 3) for _ in range(3)]
            text_input = [random.randint(-1, 3) for _ in range(3)]
            image_input = [random.randint(-1, 3) for _ in range(3)]

            emb_c, active_ids, outputs, final_output = self.forward(
                audio_input, text_input, image_input
            )

            if not active_ids:
                continue  # اگر آرایه‌ای فعال نشده، این نمونه را رد کن

            # تارگت ساده: جمع Emb-C
            target = sum(emb_c)

            # آموزش هر آرایه فعال برای نزدیک شدن به target
            total_loss = 0.0
            for arr_id in active_ids:
                arr = self.arrays[arr_id]
                pred = outputs[arr_id]
                loss = arr.update(emb_c, target, pred, lr)
                total_loss += loss

            recent_losses.append(total_loss)

            # لاگ هر 50 استپ
            if (step + 1) % 50 == 0 and recent_losses:
                avg_loss = sum(recent_losses[-50:]) / len(recent_losses[-50:])
                print(f"Step {step+1}/{steps} - avg loss: {avg_loss:.4f}")

        print("Training finished.\n")


# ==========================================================
#  RUN PHASE 3
# ==========================================================
if __name__ == "__main__":

    # 1) مدل را بساز
    model = C7Model(emb_dim=9)

    # 2) آموزش سبک
    model.train(steps=300, lr=0.01)

    # 3) یک تست با دیباگ روشن
    VERBOSE = True
    print("\n========== TEST AFTER TRAINING ==========\n")

    # ورودی تست
    audio_input = [random.randint(-1, 3) for _ in range(3)]
    text_input = [random.randint(-1, 3) for _ in range(3)]
    image_input = [random.randint(-1, 3) for _ in range(3)]

    debug("Audio Input", audio_input)
    debug("Text Input", text_input)
    debug("Image Input", image_input)

    emb_c, active_ids, outputs, final_output = model.forward(
        audio_input, text_input, image_input
    )

    target = sum(emb_c)
    print(f"\nTarget (sum(Emb-C)): {target:.4f}")
    print(f"Final Output (A7):   {final_output:.4f}")

    print("\nPer-array predictions:")
    for arr_id in sorted(outputs.keys()):
        print(f"  A{arr_id}: {outputs[arr_id]:.4f}")

    error = final_output - target
    print(f"\nFinal error (A7 - target): {error:.4f}")
    print("\n=========================================\n")