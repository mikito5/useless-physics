import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta

# ================== Time (JST) ==================
JST = timezone(timedelta(hours=9))
today = datetime.now(JST).strftime("%Y-%m-%d")
rng = np.random.default_rng()

# ================== Simulation time ==================
T = 10.0
N = 4096  # FFTしやすい
t = np.linspace(0, T, N, endpoint=False)
dt = t[1] - t[0]
fs = 1.0 / dt

# ================== Discrete component values ==================
R_VALUES = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
L_VALUES = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
C_VALUES = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0])

# ================== Input signals ==================
def make_input(kind, t, dt):
    A = rng.uniform(0.8, 1.5)
    x = np.zeros_like(t)

    if kind == "step":
        x[t > t[-1] * 0.1] = A
        desc = f"step (A={A:.2f})"

    elif kind == "impulse":
        x[int(0.1 * len(t))] = A / dt
        desc = f"impulse (area≈{A:.2f})"

    elif kind == "sine":
        f = rng.uniform(0.2, 2.0)
        x = A * np.sin(2 * np.pi * f * t)
        desc = f"sine (A={A:.2f}, f={f:.2f} Hz)"

    elif kind == "square":
        f = rng.uniform(0.2, 1.0)
        x = A * np.sign(np.sin(2 * np.pi * f * t))
        desc = f"square (A={A:.2f}, f={f:.2f} Hz)"

    elif kind == "ramp":
        x = np.clip(A * (t / t[-1]), 0, A)
        desc = f"ramp (A={A:.2f})"

    elif kind == "noise":
        x = A * rng.normal(0, 0.5, size=len(t))
        desc = f"noise (σ≈{0.5 * A:.2f})"

    else:
        raise ValueError("unknown input kind")

    return x, desc

input_kind = rng.choice(["step", "impulse", "sine", "square", "ramp", "noise"])
x, x_desc = make_input(input_kind, t, dt)

# ================== Core filters ==================
def first_order_lowpass(x, tau, dt):
    """y[n] = a y[n-1] + (1-a) x[n]"""
    y = np.zeros_like(x)
    tau = max(float(tau), 1e-9)
    a = np.exp(-dt / tau)
    b = 1.0 - a
    for n in range(1, len(x)):
        y[n] = a * y[n - 1] + b * x[n]
    return y

def rlc_analog_tf(kind, wn, zeta):
    """
    Den(s) = s^2 + 2ζωn s + ωn^2
    low:   Num(s)= ωn^2
    high:  Num(s)= s^2
    band:  Num(s)= 2ζωn s
    notch: Num(s)= s^2 + ωn^2
    Return num_s, den_s as [b0,b1,b2] for b0 s^2 + b1 s + b2
    """
    den_s = [1.0, 2.0*zeta*wn, wn**2]

    if kind == "low":
        num_s = [0.0, 0.0, wn**2]
    elif kind == "high":
        num_s = [1.0, 0.0, 0.0]
    elif kind == "band":
        num_s = [0.0, 2.0*zeta*wn, 0.0]
    elif kind == "notch":
        num_s = [1.0, 0.0, wn**2]
    else:
        raise ValueError("unknown kind")

    return num_s, den_s

def bilinear_biquad_from_analog(num_s, den_s, dt):
    """
    Analog: (b0 s^2 + b1 s + b2) / (a0 s^2 + a1 s + a2)
    Bilinear: s = K*(1 - z^-1)/(1 + z^-1), K=2/dt
    Returns b=[b0,b1,b2], a=[1,a1,a2] for DF1:
      y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2] - a1 y[n-1] - a2 y[n-2]
    """
    b0, b1, b2 = map(float, num_s)
    a0, a1, a2 = map(float, den_s)
    K = 2.0 / dt

    # (1 - z^-1), (1 + z^-1)
    p = np.array([1.0, -1.0])
    q = np.array([1.0,  1.0])

    p2 = np.convolve(p, p)   # (1 - z^-1)^2
    q2 = np.convolve(q, q)   # (1 + z^-1)^2
    pq = np.convolve(p, q)   # (1 - z^-1)(1 + z^-1) = 1 - z^-2

    B = (b0 * (K**2)) * p2 + (b1 * K) * pq + b2 * q2
    A = (a0 * (K**2)) * p2 + (a1 * K) * pq + a2 * q2

    if abs(A[0]) < 1e-18:
        A[0] = 1e-18

    B = B / A[0]
    A = A / A[0]

    return B, A  # length-3 each

def biquad_filter(x, b, a):
    """Direct Form I biquad, a[0]=1 assumed."""
    y = np.zeros_like(x)
    b0, b1, b2 = map(float, b)
    _, a1, a2 = map(float, a)

    x1 = x2 = 0.0
    y1 = y2 = 0.0
    for n in range(len(x)):
        x0 = float(x[n])
        y0 = b0*x0 + b1*x1 + b2*x2 - a1*y1 - a2*y2
        y[n] = y0
        x2, x1 = x1, x0
        y2, y1 = y1, y0
    return y

# ================== Random system + random filter-kind by system ==================
system = rng.choice(["RC", "RL", "RLC"])

if system in ["RC", "RL"]:
    filter_kind = rng.choice(["low", "high"])
else:
    filter_kind = rng.choice(["low", "high", "band", "notch"])

# ================== Simulate ==================
if system == "RC":
    R = float(rng.choice(R_VALUES))
    C = float(rng.choice(C_VALUES))
    tau = R * C

    y_lp = first_order_lowpass(x, tau, dt)
    y = y_lp if filter_kind == "low" else (x - y_lp)

    params = f"R={R:.2f}, C={C:.2f}, τ={tau:.2f}"

elif system == "RL":
    R = float(rng.choice(R_VALUES))
    L = float(rng.choice(L_VALUES))
    tau = L / max(R, 1e-12)

    y_lp = first_order_lowpass(x, tau, dt)
    y = y_lp if filter_kind == "low" else (x - y_lp)

    params = f"R={R:.2f}, L={L:.2f}, τ={tau:.2f}"

else:
    R = float(rng.choice(R_VALUES))
    L = float(rng.choice(L_VALUES))
    C = float(rng.choice(C_VALUES))

    wn = 1.0 / np.sqrt(L * C)
    zeta = (R / 2.0) * np.sqrt(C / L)
    zeta = max(zeta, 1e-6)  # safety

    num_s, den_s = rlc_analog_tf(filter_kind, wn, zeta)
    b, a = bilinear_biquad_from_analog(num_s, den_s, dt)
    y = biquad_filter(x, b, a)

    params = f"R={R:.2f}, L={L:.2f}, C={C:.2f}, ωn={wn:.2f}, ζ={zeta:.2f}"

# ================== 1) “Judgement” ==================
y_center = y - np.mean(y)
y_scale = np.max(np.abs(y_center)) + 1e-12
y_norm = y_center / y_scale

tail = y_norm[int(0.9 * len(y_norm)):]
final_level = np.mean(tail)
peak = np.max(y_norm)
overshoot = float(peak - final_level)

dy = np.diff(y_norm)
wiggles = int(np.sum((dy[:-1] * dy[1:]) < 0))

score = 0.7 * abs(overshoot) + 0.3 * (wiggles / 50.0)

if score < 0.25:
    verdict = "calm"
elif score < 0.60:
    verdict = "restless"
else:
    verdict = "feral"

# ================== 2) FFT plot ==================
x_center = x - np.mean(x)
y_center = y - np.mean(y)

window = np.hanning(len(y_center))
X = np.fft.rfft(x_center * window)
Y = np.fft.rfft(y_center * window)

freq = np.fft.rfftfreq(len(y_center), d=dt)

magX = np.abs(X)
magY = np.abs(Y)
magX[0] = 0.0
magY[0] = 0.0

mask = freq > 0
freqp = freq[mask]
magXp = magX[mask]
magYp = magY[mask]

peak_k = int(np.argmax(magYp))
peak_f = float(freqp[peak_k])
peak_mag = float(magYp[peak_k])

# Rough response-shape guess from |Y|/|X|
eps = 1e-12
H = magYp / (magXp + eps)
split = np.median(freqp)
low_med = float(np.median(H[freqp < split])) if np.any(freqp < split) else float(np.median(H))
high_med = float(np.median(H[freqp >= split])) if np.any(freqp >= split) else float(np.median(H))
lp_score = low_med / (high_med + eps)
if lp_score > 1.2:
    shape_guess = "low-ish"
elif lp_score < 1/1.2:
    shape_guess = "high-ish"
else:
    shape_guess = "band/flat-ish"

# ================== 3) One-liner ==================
one_liners = {
    "calm": [
        "Today, the system pretended to be well-behaved.",
        "Nothing happened. Therefore, it is perfect.",
        "The dynamics were suspiciously polite."
    ],
    "restless": [
        "The system wiggled, then denied everything.",
        "A small oscillation appeared and immediately regretted it.",
        "This is not chaos—just enthusiasm."
    ],
    "feral": [
        "The system chose violence (mathematically).",
        "We observed a rare event: uncontrolled confidence.",
        "A resonance-like thing happened. Please do not ask why."
    ]
}
line = rng.choice(one_liners[verdict])

# ================== Plot ==================
plt.figure(figsize=(9, 7))

ax1 = plt.subplot(2, 1, 1)
ax1.plot(t, x, label="input")
ax1.plot(t, y, label="output")
ax1.set_xlabel("t [arb.]")
ax1.set_ylabel("[arb.]")
ax1.set_title(
    f"Daily Useless {system} ({today}) | verdict: {verdict}\n"
    f"Filter: {filter_kind} | Input: {input_kind}, {x_desc} | {params}"
)
ax1.legend()
ax1.grid(True, alpha=0.25)

ax2 = plt.subplot(2, 1, 2)
ax2.plot(freqp, magXp, label="input FFT", alpha=0.6)
ax2.plot(freqp, magYp, label="output FFT")
ax2.set_xscale("log")
ax2.set_yscale("log")

xmax = min(np.max(freqp), 10.0)
ax2.set_xlim(freqp[0], xmax)

ax2.set_xlabel("frequency [arb.]")
ax2.set_ylabel("|FFT(.)|")
ax2.set_title(f"Output FFT peak ≈ {peak_f:.3f} (arb.), magnitude ≈ {peak_mag:.3e} | {shape_guess}")
ax2.grid(True, alpha=0.25)
ax2.legend()

plt.tight_layout()
plt.savefig("result.svg")
plt.close()

# ================== README ==================
readme = f"""# Daily Useless Physics

Every day, GitHub Actions generates a **random dynamical system**
and excites it with a **random input signal**.

## Today's Result ({today})

- **System**: {system}
- **Filter kind (random by system)**: {filter_kind}
- **Parameters**: {params}
- **Input**: {x_desc}

### Useless judgement
- **verdict**: **{verdict}**
- overshoot-ish: {overshoot:.3f}
- wiggles: {wiggles}
- FFT peak (output): {peak_f:.3f} (arb.)
- rough shape guess from |Y|/|X|: {shape_guess}

> {line}

![result](result.svg)
"""
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)
