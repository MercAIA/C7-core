import numpy as np

np.random.seed(42)

# ---------- Helper functions ----------

def make_random_sample():
    """ساخت ورودی رندوم برای سه modality"""
    audio = np.random.randint(-1, 4, size=3).astype(float)
    text  = np.random.randint(-1, 4, size=3).astype(float)
    image = np.random.randint(-1, 4, size=3).astype(float)
    return audio, text, image

def frontends(audio, text, image):
    """یک frontend خیلی ساده: همین بردارها رو برمی‌گردونیم (یا کمی scale می‌کنیم)."""
    A = audio.astype(float)
    T = text.astype(float)
    I = image.astype(float)
    emb_c = np.concatenate([A, T, I])
    return A, T, I, emb_c

def compute_intensity(A, T, I):
    """شدت کلی: یک عدد بین 0 و حدوداً 1"""
    ea = np.linalg.norm(A)
    et = np.linalg.norm(T)
    ei = np.linalg.norm(I)
    s = ea + et + ei
    return np.tanh(0.15 * s)

def simulate_user_feedback(base_err):
    """
    فیدبک مصنوعی کاربر:
    - خطای خیلی کم → کم پیش میاد غر بزند
    - خطای متوسط → گاهی ناراضی
    - خطای زیاد → اغلب ناراضی
    """
    if base_err < 0.5:
        p_bad = 0.05
    elif base_err < 2.0:
        p_bad = 0.30
    else:
        p_bad = 0.80
    return 1 if np.random.rand() < p_bad else 0

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------- Model parameters ----------

# شلو / دیپ: هر دو روی مجموع Emb-C یک خطی ساده می‌زنن
w_sh = 0.5
b_sh = 0.0

w_dp = -0.3
b_dp = 0.0

# گیت: ورودی‌اش [1, norm_err, intensity] است
w_gate = np.array([0.0, 0.0, 0.0], dtype=float)

# تنظیمات آموزش
num_steps = 800
lr_main  = 1e-3
lr_gate  = 1e-3

ema_err = 1.0          # میانگین نمایی خطای شلو
self_thr = 0.7         # زیر این، مدل خودش را "خوب" فرض می‌کند

# ---------- Training loop ----------

print("Starting Phase 9 (surprise + user_feedback gate) training for {} steps...\n".format(num_steps))

for step in range(1, num_steps + 1):
    # 1) نمونه رندوم
    audio, text, image = make_random_sample()
    A, T, I, emb_c = frontends(audio, text, image)
    emb_sum = float(np.sum(emb_c))
    target  = float(np.sum(emb_c))  # مثل قبل: target = sum(Emb-C)
    intensity = compute_intensity(A, T, I)

    # 2) خروجی شلو و دیپ
    y_sh = w_sh * emb_sum + b_sh
    y_dp = w_dp * emb_sum + b_dp

    base_err = float(abs(y_sh - target))

    # آپدیت میانگین خطا
    ema_err = 0.99 * ema_err + 0.01 * base_err

    # 3) سیگنال‌های درونی و بیرونی
    self_good = 1.0 if base_err < self_thr else 0.0
    user_bad  = float(simulate_user_feedback(base_err))

    # خطای نرمال‌شده نسبت به وضعیت کلی اخیر
    norm_err = base_err / (1e-3 + ema_err)

    # 4) ورودی گیت (فقط چیا رو می‌بینه؟ norm_err + intensity)
    x_gate = np.array([1.0, norm_err, intensity], dtype=float)
    z_gate = float(np.dot(w_gate, x_gate))
    g      = float(sigmoid(z_gate))

    # 5) خروجی نهایی
    y_final = (1.0 - g) * y_sh + g * y_dp
    err     = y_final - target
    loss_main = err ** 2

    # 6) هدف گیت: اگر user_bad=1 → گیت باید به سمت 1 متمایل شود
    # اگر user_bad=0 → گیت باید نرمال و نزدیک 0 بماند
    gate_target = user_bad
    loss_gate = (g - gate_target) ** 2

    loss = loss_main + 0.1 * loss_gate   # کمی وزن برای بخش گیت

    # ---------- Gradients ----------

    # dL/dy_final
    dL_dy = 2.0 * err

    # مشتقات نسبت به y_sh و y_dp
    dL_dy_sh = dL_dy * (1.0 - g)
    dL_dy_dp = dL_dy * g

    # شلو
    dL_dw_sh = dL_dy_sh * emb_sum
    dL_db_sh = dL_dy_sh

    # دیپ
    dL_dw_dp = dL_dy_dp * emb_sum
    dL_db_dp = dL_dy_dp

    # فقط از loss_gate برای آپدیت گیت استفاده می‌کنیم (ساده‌تر و تمیزتر)
    dL_dg_gate = 2.0 * (g - gate_target)
    dg_dz      = g * (1.0 - g)
    dL_dz_gate = dL_dg_gate * dg_dz

    dL_dw_gate = dL_dz_gate * x_gate  # چون z = w·x

    # ---------- Parameter update ----------

    w_sh -= lr_main * dL_dw_sh
    b_sh -= lr_main * dL_db_sh

    w_dp -= lr_main * dL_dw_dp
    b_dp -= lr_main * dL_db_dp

    w_gate -= lr_gate * dL_dw_gate

    # ---------- Logging ----------

    if step % 50 == 0:
        print(
            f"Step {step:4d} | loss:{loss:7.3f} err:{err:7.3f} "
            f"base_err:{base_err:5.3f} g:{g:5.3f} "
            f"norm_err:{norm_err:5.3f} user_bad:{int(user_bad)} self_good:{int(self_good)} int:{intensity:5.3f}"
        )

print("\nTraining finished.\n")

# ---------- EVAL on fixed sample ----------

print("========== FIXED SAMPLE EVAL ==========")
audio = np.array([2.0, 1.0, 2.0])
text  = np.array([3.0, 3.0, 3.0])
image = np.array([-1.0, 0.0, 1.0])

A, T, I, emb_c = frontends(audio, text, image)
emb_sum = float(np.sum(emb_c))
target  = float(np.sum(emb_c))
intensity = compute_intensity(A, T, I)

y_sh = w_sh * emb_sum + b_sh
y_dp = w_dp * emb_sum + b_dp
base_err = float(abs(y_sh - target))
norm_err = base_err / (1e-3 + ema_err)

x_gate = np.array([1.0, norm_err, intensity], dtype=float)
z_gate = float(np.dot(w_gate, x_gate))
g      = float(sigmoid(z_gate))
y_final = (1.0 - g) * y_sh + g * y_dp
err     = y_final - target

print("Emb-C Vector       :", emb_c.tolist())
print(f"Target (sum Emb-C) : {target:7.3f}")
print(f"Intensity          : {intensity:7.3f}")
print(f"y_shallow          : {y_sh:7.3f}")
print(f"y_deep             : {y_dp:7.3f}")
print(f"gate g             : {g:7.3f}")
print(f"y_final            : {y_final:7.3f}")
print(f"Final error        : {err:7.3f}")
print("========================================\n")

# ---------- RANDOM EVAL ----------

print("========== RANDOM EVAL (10 samples) ==========")
abs_errs = []
g_list   = []
for i in range(10):
    audio, text, image = make_random_sample()
    A, T, I, emb_c = frontends(audio, text, image)
    emb_sum = float(np.sum(emb_c))
    target  = float(np.sum(emb_c))
    intensity = compute_intensity(A, T, I)

    y_sh = w_sh * emb_sum + b_sh
    y_dp = w_dp * emb_sum + b_dp
    base_err = float(abs(y_sh - target))
    norm_err = base_err / (1e-3 + ema_err)

    x_gate = np.array([1.0, norm_err, intensity], dtype=float)
    z_gate = float(np.dot(w_gate, x_gate))
    g      = float(sigmoid(z_gate))
    y_final = (1.0 - g) * y_sh + g * y_dp
    err     = y_final - target

    abs_errs.append(abs(err))
    g_list.append(g)

    mode = "DEEP " if g > 0.5 else "SHALL"
    print(
        f"# {i+1:2d} | int:{intensity:5.3f} base_err:{base_err:6.3f} "
        f"g:{g:5.3f} mode:{mode} target:{target:6.2f} y:{y_final:6.2f} abs_err:{abs(err):6.3f}"
    )

print("----------------------------------------------")
print(f"Mean |error| over 10 samples : {np.mean(abs_errs):6.3f}")
print(f"Mean gate g                  : {np.mean(g_list):6.3f}")
print("==============================================")