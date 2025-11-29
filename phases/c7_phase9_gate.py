import numpy as np

# ========== SETTINGS ==========
num_steps = 300

# ========== SIMPLE SHALLOW MODEL ==========
def shallow_model(x):
    return np.sum(x) * 0.7

# ========== SIMPLE DEEP MODEL ==========
def deep_model(x):
    return np.sum(x) * 1.05 + 3.0

# ========== INTENSITY FUNCTION ==========
def compute_intensity(x):
    return float(np.mean(np.abs(x)))

# ========== COHERENCE FUNCTION ==========
def compute_coherence(x):
    v = np.var(x)
    return 1 / (1 + v)

# ========== SURVIVAL CORE ==========
def survival_core(err_history):
    if len(err_history) == 0:
        return 1.0
    return float(1 + np.mean(np.abs(err_history)) * 0.3)

# ========== GATE FUNCTION (GROUNDED) ==========
def gate_function(intensity, coherence, SC):
    # grounding prevents "always deep"
    threshold = 0.55 * SC  
    return 1.0 if intensity < threshold else 0.0


# ========== TRAINING LOOP ==========
err_history = []

for step in range(1, num_steps + 1):

    x = np.random.uniform(-1, 3, size=9)
    target = float(np.sum(x))

    # shallow + deep
    y_shallow = shallow_model(x)
    y_deep = deep_model(x)

    # metrics
    intensity = compute_intensity(x)
    coherence = compute_coherence(x)
    SC = survival_core(err_history)

    # conditional grounded gate
    g = gate_function(intensity, coherence, SC)

    # final output
    y = g * y_deep + (1 - g) * y_shallow
    err = float(y - target)

    err_history.append(err)

    if step % 50 == 0:
        print(f"Step {step:4d} | err:{err:+.3f}  |g:{g:.3f} "
              f" int:{intensity:.3f} coh:{coherence:.3f} SC:{SC:.3f}")

print("\nTraining finished.\n")