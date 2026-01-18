import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta

# ================== Time (JST) ==================
JST = timezone(timedelta(hours=9))
today = datetime.now(JST).strftime("%Y-%m-%d")
rng = np.random.default_rng()

# ================== Simulation time ==================
T = 10.0
N = 4096  # FFTしやすいように少し大きめ
t = np.linspace(0, T, N)
dt = t[1] - t[0]
fs = 1.0 / dt  # [1/s] (tの単位が"秒"という建て付け)

# ================== Input signals ==================
def make_input(kind, t):
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
x, x_desc = make_input(input_kind, t)

# ================== System selection ==================
system = rng.choice(["RC", "RL", "RLC"])

# ---------- RC / RL: 1st order low-pass form ----------
def first_order_response(x, tau, dt):
    y = np.zeros_like(x)
    a = np.exp(-dt / tau)
    b = 1.0 - a
    for n in range(1, len(x)):
        y[n] = a * y[n - 1] + b * x[n]
    return y

# ---------- Simulate ----------
if system == "RC":
    R = rng.uniform(0.5, 5.0)
    C = rng.uniform(0.2, 5.0)
    tau = R * C
    y = first_order_response(x, tau, dt)
    params = f"R={R:.2f}, C={C:.2f}, τ={tau:.2f}"

elif system == "RL":
    R = rng.uniform(0.5, 5.0)
    L = rng.uniform(0.5, 5.0)
    tau = L / R
    y = first_order_response(x, tau, dt)
    params = f"R={R:.2f}, L={L:.2f}, τ={tau:.2f}"

else:
    # Standard 2nd order form: y'' + 2ζωn y' + ωn^2 y = ωn^2 x
    R = rng.uniform(0.3, 2.0)
    L = rng.uniform(0.5, 3.0)
    C = rng.uniform(0.2, 3.0)
    wn = 1.0 / np.sqrt(L * C)
    zeta = (R / 2.0) * np.sqrt(C / L)

    y = np.zeros_like(x)
    yd = np.zeros_like(x)  # y'
    for n in range(1, len(x)):
        ydd = (wn ** 2) * (x[n] - y[n - 1]) - (2 * zeta * wn) * yd[n - 1]
        yd[n] = yd[n - 1] + dt * ydd
        y[n] = y[n - 1] + dt * yd[n]

    params = f"R={R:.2f}, L={L:.2f}, C={C:.2f}, ωn={wn:.2f}, ζ={zeta:.2f}"

# ================== 1) “Judgement” (meaningless but consistent) ==================
# Normalize for analysis stability
y_center = y - np.mean(y)
y_scale = np.max(np.abs(y_center)) + 1e-12
y_norm = y_center / y_scale

# Overshoot-ish metric (peak compared to final-ish level)
tail = y_norm[int(0.9 * len(y_norm)):]
final_level = np.mean(tail)
peak = np.max(y_norm)
overshoot = float(peak - final_level)

# Oscillation metric: count sign changes of derivative (rough “wiggle count”)
dy = np.diff(y_norm)
wiggles = int(np.sum((dy[:-1] * dy[1:]) < 0))  # derivative sign changes

# “Stability” score (completely arbitrary but repeatable)
score = 0.7 * abs(overshoot) + 0.3 * (wiggles / 50.0)

if score < 0.25:
    verdict = "calm"
elif score < 0.60:
    verdict = "restless"
else:
    verdict = "feral"

# ================== 2) FFT plot (output spectrum) ==================
# Windowing to make spectrum nicer
window = np.hanning(len(y_center))
Y = np.fft.rfft(y_center * window)
freq = np.fft.rfftfreq(len(y_center), d=dt)
mag = np.abs(Y)

# Avoid the DC bin dominating the plot
mag[0] = 0.0
mask = freq > 0
freqp = freq[mask]
magp = mag[mask]
peak_k = int(np.argmax(mag))
peak_f = float(freq[peak_k])
peak_mag = float(mag[peak_k])

# ================== 3) English one-liner ==================
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

# ================== Plot (time + FFT) ==================
plt.figure(figsize=(9, 7))

# Top: time domain
ax1 = plt.subplot(2, 1, 1)
ax1.plot(t, x, label="input")
ax1.plot(t, y, label="output")
ax1.set_xlabel("t [arb.]")
ax1.set_ylabel("[arb.]")
ax1.set_title(
    f"Daily Useless {system} ({today}) | verdict: {verdict} ({verdict_jp})\n"
    f"Input: {input_kind}, {x_desc} | {params}"
)
ax1.legend()
ax1.grid(True, alpha=0.25)

# Bottom: spectrum
ax2 = plt.subplot(2, 1, 2)
ax2.plot(freqp, magp, label="output FFT")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlim(0, min(np.max(freq), 10.0))  # 見やすさ優先で低周波側だけ
ax2.set_xlabel("frequency [arb.]")
ax2.set_ylabel("|FFT(output)|")
ax2.set_title(f"FFT peak ≈ {peak_f:.3f} (arb.), magnitude ≈ {peak_mag:.3e}")
ax2.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig("result.svg")
plt.close()

# ================== README ==================
readme = f"""# Daily Useless Physics

Every day, GitHub Actions generates a **random dynamical system**
and excites it with a **random input signal**.

## Today's Result ({today})

- **System**: {system}
- **Parameters**: {params}
- **Input**: {x_desc}

### Useless judgement
- **verdict**: **{verdict}**（{verdict_jp}）
- overshoot-ish: {overshoot:.3f}
- wiggles: {wiggles}
- FFT peak: {peak_f:.3f} (arb.)

> {line}

![result](result.svg)

"""
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)
