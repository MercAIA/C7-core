import numpy as np

# =========================
#  C7 Phase 9 - Gated Deep/Shallow
#  - Emb-C از سه فرانت‌اند
#  - سر شالو (y_shallow)
#  - سر دیپ ساده (y_deep)
#  - گیت g که یاد می‌گیرد کی دیپ لازم است
#  - خروجی نهایی: y = (1-g)*y_shallow + g*y_deep
#  - هدف: sum(Emb-C)
# =========================

rng = np.random.RandomState(42)

# ---------- Frontends (فیکس) ----------

def audio_frontend(a):
    a = np.array(a, dtype=float)
    # همون الگوی قبلی: کمی scale + شیفت
    return np.array([
        a[0] + 1.0,
        0.5 * (a[0] + a[1]),
        a[2] + 1.0
    ], dtype=float)

def text_frontend(t):
    t = np.array(t, dtype=float)
    return np.array([
        3.0 * t[0],
        3.0 * t[1] + 1.0,
        3.0 * t[2] - 1.0
    ], dtype=float)

def image_frontend(i):
    i = np.array(i, dtype=float)
    return np.array([
        0.5 * i[0],
        1.0 * i[1],
        0.5 * (i[1] + i[2])
    ], dtype=float)

def make_emb(audio, text, image):
    A = audio_frontend(audio)
    T = text_frontend(text)
    I = image_frontend(image)
    emb = np.concatenate([A, T, I]).astype(float)
    return emb


# ---------- مدل بالا دستی: شالو + دیپ + گیت ----------

dim_emb = 9

# سر شالو: خطی
w_s = rng.randn(dim_emb) * 0.05
b_s = 0.0

# سر دیپ: خطی جداگانه (می‌تونه وزن‌های متفاوت یاد بگیره)
w_d = rng.randn(dim_emb) * 0.05
b_d = 0.0

# گیت: ورودی‌اش [intensity, base_err_norm]
w_g = rng.randn(2) * 0.05
b_g = 0.0

# self-image ساده: یه تخمین از خطای معمول شالو
running_base_err = 1.0

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def forward_once(audio, text, image, target=None):
    """
    یک پاس فوروارد:
    - emb, intensity
    - y_shallow, y_deep
    - g، y_final
    اگر target داده شود، base_err هم حساب می‌شود.
    """
    emb = make_emb(audio, text, image)

    # شدت (intensity) مثل قبل: نرمال شده
    norm = np.linalg.norm(emb)
    intensity = norm / (1.0 + norm)

    # شالو
    y_shallow = float(np.dot(w_s, emb) + b_s)

    # دیپ (خطی جداگانه؛ از لحاظ نقش "دیپ‌تر" است)
    y_deep = float(np.dot(w_d, emb) + b_d)

    # خطای شالو (فقط وقتی target داریم)
    if target is not None:
        base_err = abs(y_shallow - target)
    else:
        base_err = None

    return emb, intensity, y_shallow, y_deep, base_err


# ---------- آموزش ----------

num_steps = 800
lr = 0.0005

# وزن cost استفاده از دیپ (gate_penalty): دیپ ارزان نیست!
lambda_gate = 0.01

print("Starting Phase 9 (gated shallow/deep) training for", num_steps, "steps...\n")

for step in range(1, num_steps + 1):
    # ورودی تصادفی شبیه فازهای قبل
    audio = rng.randint(-1, 4, size=3)
    text  = rng.randint(-1, 4, size=3)
    image = rng.randint(-1, 4, size=3)

    emb, intensity, y_shallow, y_deep, _ = forward_once(audio, text, image)

    # هدف: sum(Emb-C)
    target = float(emb.sum())

    # خطای شالو برای گیت
    base_err = abs(y_shallow - target)

    # self-image: تخمین نرم از خطای شالو
    running_base_err = 0.98 * running_base_err + 0.02 * base_err

    # فیچرهای گیت: شدت و base_err نرمال‌شده
    base_err_norm = base_err / (1.0 + running_base_err)
    gate_in = np.array([intensity, base_err_norm], dtype=float)

    z = float(np.dot(w_g, gate_in) + b_g)
    g = sigmoid(z)  # بین ۰ و ۱

    # خروجی نهایی
    # y = y_s + g*(y_d - y_s)
    y = y_shallow + g * (y_deep - y_shallow)

    # لا‌س
    err = y - target
    loss_main = err * err
    loss_gate = lambda_gate * (g * g)     # استفاده زیاد از دیپ، هزینه دارد
    loss = loss_main + loss_gate

    # --------- گرادیان‌ها (دست‌ساز) ---------

    # dL/dy
    dL_dy = 2.0 * err

    # y = y_s + g*(y_d - y_s)
    # ∂y/∂y_s = 1 - g
    # ∂y/∂y_d = g
    # ∂y/∂g   = (y_d - y_s)
    dL_dy_s = dL_dy * (1.0 - g)
    dL_dy_d = dL_dy * g
    dL_dg_main = dL_dy * (y_deep - y_shallow)

    # + مشتق term گیت
    dL_dg_gate = 2.0 * lambda_gate * g
    dL_dg = dL_dg_main + dL_dg_gate

    # سر شالو: y_s = w_s · emb + b_s
    dL_dw_s = dL_dy_s * emb
    dL_db_s = dL_dy_s

    # سر دیپ: y_d = w_d · emb + b_d
    dL_dw_d = dL_dy_d * emb
    dL_db_d = dL_dy_d

    # گیت: g = sigmoid(z), z = w_g · gate_in + b_g
    # ∂g/∂z = g*(1-g)
    dL_dz = dL_dg * g * (1.0 - g)
    dL_dw_g = dL_dz * gate_in
    dL_db_g = dL_dz

    # --------- آپدیت پارامترها ---------
    w_s -= lr * dL_dw_s
    b_s -= lr * dL_db_s

    w_d -= lr * dL_dw_d
    b_d -= lr * dL_db_d

    w_g -= lr * dL_dw_g
    b_g -= lr * dL_db_g

    # لاگ هر ۵۰ استپ
    if step % 50 == 0:
        print(
            f"Step {step:4d} | "
            f"loss:{loss:7.3f} err:{err:7.3f} "
            f"int:{intensity:4.3f} base_err:{base_err:5.3f} "
            f"g:{g:4.3f} baseErrAvg:{running_base_err:5.3f}"
        )

print("\nTraining finished.\n")

# ---------- تست روی نمونه ثابت (مثل همیشه) ----------

test_audio = np.array([2, 1, 2])
test_text  = np.array([3, 3, 3])
test_image = np.array([-1, 0, 1])

emb, intensity, y_shallow, y_deep, _ = forward_once(test_audio, test_text, test_image)
target = float(emb.sum())
base_err = abs(y_shallow - target)
base_err_norm = base_err / (1.0 + running_base_err)
gate_in = np.array([intensity, base_err_norm], dtype=float)
z = float(np.dot(w_g, gate_in) + b_g)
g = sigmoid(z)
y = y_shallow + g * (y_deep - y_shallow)
final_err = y - target

print("========== FIXED SAMPLE EVAL ==========")
print("Audio Input :", test_audio.tolist())
print("Text  Input :", test_text.tolist())
print("Image Input :", test_image.tolist())
print()
print("Emb-C Vector        :", np.round(emb, 3).tolist())
print(f"Target (sum Emb-C)  : {target:7.3f}")
print(f"Intensity           : {intensity:7.3f}")
print(f"Base error (shallow): {base_err:7.3f}")
print()
print(f"y_shallow           : {y_shallow:7.3f}")
print(f"y_deep              : {y_deep:7.3f}")
print(f"gate g              : {g:7.3f}")
print(f"y_final             : {y:7.3f}")
print(f"Final error         : {final_err:7.3f}")
print("======================================\n")

# ---------- چند نمونه رندوم برای دیدن رفتار گیت ----------

print("========== RANDOM EVAL (10 samples) ==========")
abs_errs = []
g_list = []
for i in range(10):
    audio = rng.randint(-1, 4, size=3)
    text  = rng.randint(-1, 4, size=3)
    image = rng.randint(-1, 4, size=3)

    emb, intensity, y_s, y_d, _ = forward_once(audio, text, image)
    target = float(emb.sum())
    base_err = abs(y_s - target)
    base_err_norm = base_err / (1.0 + running_base_err)
    gate_in = np.array([intensity, base_err_norm], dtype=float)
    z = float(np.dot(w_g, gate_in) + b_g)
    g = sigmoid(z)
    y = y_s + g * (y_d - y_s)
    err = y - target

    abs_errs.append(abs(err))
    g_list.append(g)

    mode = "DEEP" if g > 0.5 else "SHALLOW"
    print(
        f"#{i+1:2d} | int:{intensity:4.3f} base_err:{base_err:5.3f} "
        f"g:{g:4.3f} mode:{mode:7s} target:{target:6.2f} "
        f"y:{y:6.2f} abs_err:{abs(err):5.3f}"
    )

print("----------------------------------------------")
print(f"Mean |error| over 10 samples : {np.mean(abs_errs):5.3f}")
print(f"Mean gate g                  : {np.mean(g_list):5.3f}")
print("==============================================")