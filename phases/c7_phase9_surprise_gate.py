import numpy as np

# ---------------------------------------------------------
# Phase 9 – Surprise-based gating between SHALLOW and DEEP
# ---------------------------------------------------------

rng = np.random.RandomState(42)

def random_triplet():
    """Random toy inputs for audio / text / image."""
    # values in [-1, 0, 1, 2, 3]
    A = rng.randint(-1, 4, size=3)
    T = rng.randint(-1, 4, size=3)
    I = rng.randint(-1, 4, size=3)
    return A.astype(float), T.astype(float), I.astype(float)

def emb_collapse(A, T, I):
    """Collapse into Emb-C (simple concat here)."""
    return np.concatenate([A, T, I])

def compute_intensity(A, T, I):
    """Same سبک قبلی: dominant energy / total energy."""
    e_a = np.linalg.norm(A)
    e_t = np.linalg.norm(T)
    e_i = np.linalg.norm(I)
    total = e_a + e_t + e_i + 1e-8
    dom = max(e_a, e_t, e_i)
    return dom / total, {"audio": e_a, "text": e_t, "image": e_i}

# hyperparams
dim = 9
lr = 1e-3
num_steps = 800

# internal thresholds (self vs user)
SELF_TOL = 0.40   # مدل از نظر خودش «اوکی» است اگر norm_err < SELF_TOL
USER_TOL = 0.20   # دنیا / کاربر از نظر ما «حساس‌تر» است

# weights for shallow / deep
W_shallow = 0.01 * rng.randn(dim)
b_shallow = 0.0

W_deep = 0.01 * rng.randn(dim)
b_deep = 0.0

def surprise_gate(base_err):
    """
    base_err: |y_shallow - target|
    اینجا خود feedback_bad را از همین خطا می‌سازیم
    ولی طوری که «کاربر سختگیرتر از خود مدل باشد».
    """
    # normalize error into ~[0,1]
    norm_err = np.tanh(0.1 * base_err)  # scale 0.1 فقط برای نرم‌تر شدن

    # self view
    self_good = norm_err < SELF_TOL
    # world / user view
    user_bad = norm_err > USER_TOL   # کاربر وقتی حساس‌تر است

    if user_bad:
        if self_good:
            # حالت جالب: «من فکر می‌کردم خوبه ولی دنیا می‌گه بده»
            # اینجا surprise خیلی بالا
            surprise = 1.0 - norm_err  # هرچه norm_err کوچک‌تر، سورپرایز بزرگ‌تر
        else:
            # هم خودم می‌دونم بدم، هم دنیا می‌گه بَده → تعجب کمتر
            # ولی هنوز یک درایو برای دیپ داریم
            surprise = 0.3 * (norm_err - USER_TOL) / max(1e-6, (1.0 - USER_TOL))
            surprise = max(0.0, min(1.0, surprise))
    else:
        # دنیا هم راضی‌ست → سورپرایز صفر
        surprise = 0.0

    # ترکیب یک درایو از «خود خطا» + «سورپرایز»
    drive = 0.4 * norm_err + 0.6 * surprise
    g = np.clip(drive, 0.0, 1.0)

    return g, norm_err, surprise, int(user_bad), int(self_good)

print("Starting Phase 9 (surprise–gate) training for", num_steps, "steps...\n")

for step in range(1, num_steps + 1):
    # 1) sample
    A, T, I = random_triplet()
    emb = emb_collapse(A, T, I)
    target = emb.sum()

    # 2) predictions
    y_shallow = float(np.dot(W_shallow, emb) + b_shallow)
    y_deep    = float(np.dot(W_deep, emb)    + b_deep)

    base_err = abs(y_shallow - target)

    # 3) gate based on surprise
    g, norm_err, surprise, user_bad, self_good = surprise_gate(base_err)

    # 4) final output
    y_final = (1.0 - g) * y_shallow + g * y_deep
    err = y_final - target
    loss = 0.5 * err * err

    # 5) gradients (simple linear model)
    dy = err
    dy_shallow = dy * (1.0 - g)
    dy_deep    = dy * g

    W_shallow -= lr * dy_shallow * emb
    b_shallow -= lr * dy_shallow

    W_deep    -= lr * dy_deep * emb
    b_deep    -= lr * dy_deep

    if step % 50 == 0:
        intensity, _ = compute_intensity(A, T, I)
        print(
            f"Step {step:4d} | loss:{loss:7.3f} "
            f"err:{err:6.3f} base_err:{base_err:6.3f} "
            f"g:{g:5.3f} norm_err:{norm_err:5.3f} "
            f"surprise:{surprise:5.3f} user_bad:{user_bad} self_good:{self_good} "
            f"int:{intensity:5.3f}"
        )

print("\nTraining finished.\n")

# ---------------------------------------------------------
# EVAL on fixed sample (همون ورودی کلاسیک ما)
# ---------------------------------------------------------
A = np.array([2.0, 1.0, 2.0])
T = np.array([3.0, 3.0, 3.0])
I = np.array([-1.0, 0.0, 1.0])

emb = emb_collapse(A, T, I)
target = emb.sum()
intensity, energies = compute_intensity(A, T, I)

y_shallow = float(np.dot(W_shallow, emb) + b_shallow)
y_deep    = float(np.dot(W_deep, emb)    + b_deep)
base_err  = abs(y_shallow - target)

g, norm_err, surprise, user_bad, self_good = surprise_gate(base_err)
y_final = (1.0 - g) * y_shallow + g * y_deep
final_err = y_final - target

print("========== FIXED SAMPLE EVAL ==========")
print("Audio Input :", A.tolist())
print("Text  Input :", T.tolist())
print("Image Input :", I.tolist())
print()
print("Emb-C Vector       :", emb.tolist())
print(f"Target (sum Emb-C) : {target:7.3f}")
print()
print(f"Intensity          : {intensity:7.3f}")
print("Energies           :", {k: float(v) for k, v in energies.items()})
print()
print(f"y_shallow          : {y_shallow:7.3f}")
print(f"y_deep             : {y_deep:7.3f}")
print(f"gate g             : {g:7.3f}")
print(f"norm_err           : {norm_err:7.3f}")
print(f"surprise           : {surprise:7.3f}")
print(f"user_bad           : {user_bad}")
print(f"self_good          : {self_good}")
print(f"y_final            : {y_final:7.3f}")
print(f"Final error        : {final_err:7.3f}")
print("========================================")
print()

# ---------------------------------------------------------
# RANDOM EVAL
# ---------------------------------------------------------
print("========== RANDOM EVAL (10 samples) ==========")
abs_errs = []
gates = []
for i in range(10):
    A, T, I = random_triplet()
    emb = emb_collapse(A, T, I)
    target = emb.sum()
    intensity, _ = compute_intensity(A, T, I)

    y_shallow = float(np.dot(W_shallow, emb) + b_shallow)
    y_deep    = float(np.dot(W_deep, emb)    + b_deep)
    base_err  = abs(y_shallow - target)

    g, norm_err, surprise, user_bad, self_good = surprise_gate(base_err)
    y_final = (1.0 - g) * y_shallow + g * y_deep
    err = y_final - target

    abs_errs.append(abs(err))
    gates.append(g)

    mode = "DEEP" if g > 0.5 else ("MIX" if g > 0.1 else "SHALLOW")

    print(
        f"#{i+1:2d} | int:{intensity:5.3f} "
        f"base_err:{base_err:6.3f} g:{g:5.3f} mode:{mode:7s} "
        f"user_bad:{user_bad} self_good:{self_good} "
        f"surprise:{surprise:5.3f} target:{target:6.2f} y:{y_final:7.3f} "
        f"abs_err:{abs(err):6.3f}"
    )

print("----------------------------------------------")
print(f"Mean |error| over 10 samples : {np.mean(abs_errs):6.3f}")
print(f"Mean gate g                  : {np.mean(gates):6.3f}")
print("==============================================")