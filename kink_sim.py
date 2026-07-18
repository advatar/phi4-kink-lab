#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kink_sim.py — Relativistic inertia from pure field energy: the phi^4 kink.
===========================================================================

THE CLAIM WE ARE TESTING
------------------------
Take a single real scalar field phi(x,t) in one space dimension, with the
Lagrangian density (natural units, c = 1):

    L = 1/2 phi_t^2 - 1/2 phi_x^2 - V(phi),    V(phi) = (lambda/4)(phi^2 - v^2)^2

with lambda = 2, v = 1, so V(phi) = 1/2 (phi^2 - 1)^2.

There is NO mass parameter anywhere in this theory — only a field and a
double-well potential with two vacua at phi = -1 and phi = +1.  Yet the
theory contains a particle-like object: the "kink", a smooth wall of field
that interpolates between the two vacua,

    phi_K(x) = tanh(x)          (width w = 1 in these units).

The kink cannot decay: unwinding it would require flipping the field over
an entire half-line, at infinite energy cost.  Its energy is finite and
computable in closed form:

    M = (2*sqrt(2)/3) * sqrt(lambda) * v^3 = 4/3    (exactly, for lambda=2, v=1)

If trapped field energy really behaves like inertial mass, then a kink
moving at speed u must satisfy the relativistic point-particle relations

    E(u) = gamma * M,      P(u) = gamma * M * u,      gamma = 1/sqrt(1 - u^2),

and its width must Lorentz-contract to w/gamma.  Nothing forces this by
hand: we simply integrate the field equation and *measure* E, P, speed,
and width.  The relativity comes out of the wave equation itself (the
d'Alembertian phi_tt - phi_xx is Lorentz invariant), not from any inserted
particle mechanics.

THE EQUATION OF MOTION (mind the sign!)
---------------------------------------
    phi_tt = phi_xx - dV/dphi = phi_xx - lambda*phi*(phi^2 - v^2)
           = phi_xx - 2*phi*(phi^2 - 1)

Sanity check that tanh is a static solution:
    d^2/dx^2 tanh(x) = -2 tanh(x) sech^2(x) = -2 tanh(x)(1 - tanh^2(x))
                     = +2 phi (phi^2 - 1)   with phi = tanh(x)
so phi_xx - 2 phi(phi^2 - 1) = 0.  Good.  The vacua phi = +-1 are stable
(V'' = 6 phi^2 - 2 = 4 > 0 there; small ripples are massive waves, m = 2).

THE BOOSTED KINK (exact solution, used only as INITIAL data)
------------------------------------------------------------
Because the field equation is Lorentz covariant, boosting the static kink
gives another exact solution:

    phi(x,t)  = tanh(gamma*(x - x0 - u*t))
    phi(x,0)  = tanh(gamma*(x - x0))
    phi_t(x,0)= -u*gamma*sech^2(gamma*(x - x0))     [chain rule: d/dt of the
                argument is -u*gamma, and tanh' = sech^2 — hence the minus]

We hand the solver ONLY the t=0 data; everything after that is honest
numerical evolution of the nonlinear PDE.

WHAT WE MEASURE (all from the fields, no particle concepts inserted)
--------------------------------------------------------------------
    energy density   e(x) = 1/2 phi_t^2 + 1/2 phi_x^2 + 1/2 (phi^2-1)^2
    total energy     E    = integral e dx            -> expect gamma*M
    field momentum   P    = -integral phi_t phi_x dx -> expect gamma*M*u
    kink center      x_c(t): zero crossing of phi (linear interpolation)
                     -> straight line; fitted slope must equal u
    kink width       b: least-squares fit of tanh((x-a)/b) at the final
                     time -> expect 1/gamma (Lorentz contraction).  Note
                     the exact slope at the center is phi_x = gamma.

EXPERIMENTS
-----------
  A. Velocity sweep u in {0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95}: E, P, speed,
     width vs the relativistic predictions.
  B. Energy-conservation detail for u = 0.8 (relative drift vs time).
  C. Stability: whack the static kink with a Gaussian pulse; the kink must
     shrug it off (solitons are attractors of their topological sector).
  D. Kink–antikink collision at u = +-0.5.  Since 0.5 exceeds the known
     phi^4 critical velocity v_c ~ 0.26, the pair should bounce once and
     re-separate (below v_c they would typically capture into a "bion").

NUMERICS
--------
  * Grid x in [-30, 130], dx = 0.025 (6401 points).  dt = 0.01, so the CFL
    number dt/dx = 0.4 is comfortably below 1.
  * Time stepping: velocity-Verlet (leapfrog) — symplectic, time-reversible,
    second order, with bounded (not growing) energy error.
  * Space: centered second differences for phi_xx in the equation of motion;
    centered first differences for phi_x in the energy/momentum integrals.
  * Boundaries: Dirichlet — the two end values are pinned to their initial
    values.  All measurements are made with the lump(s) at least 20 units
    from any boundary.
  * The stability run (experiment C) uses its own grid x in [-45, 80],
    chosen so that even near-luminal radiation from the poke cannot reflect
    off a wall and re-enter the width-fit window |x| <= 10 before t_final:
    the left-wall round trip is 50 + 35 = 85 > 60, and the right wall
    (distance 75) is not reached at all.  The run is strictly
    reflection-free where it is measured.
  * Everything is deterministic; no random numbers are used.

Run:  python kink_sim.py       (writes ./out/*.png, ./out/results.json,
                                ./out/curves.json; prints a summary table)
"""

import json
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless rendering, no display needed
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

T_START = time.time()

# ----------------------------------------------------------------------
# 0. Model constants and output locations
# ----------------------------------------------------------------------
LAM = 2.0          # quartic coupling lambda
VEV = 1.0          # vacuum expectation value v
M_KINK = (2.0 * np.sqrt(2.0) / 3.0) * np.sqrt(LAM) * VEV**3   # = 4/3 exactly
assert abs(M_KINK - 4.0 / 3.0) < 1e-14

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Numerical core
# ----------------------------------------------------------------------

def accel(phi, dx):
    """Right-hand side of phi_tt = phi_xx - 2*phi*(phi^2 - 1).

    Centered second difference for phi_xx.  The two boundary entries are
    left at zero: together with pi = 0 there, this pins the end values of
    phi forever (Dirichlet boundary conditions).
    """
    a = np.zeros_like(phi)
    core = phi[1:-1]
    a[1:-1] = (phi[2:] - 2.0 * core + phi[:-2]) / dx**2 \
        - LAM * core * (core**2 - VEV**2)
    return a


def evolve(phi, pi, dx, dt, n_steps, sample_every, on_sample):
    """Velocity-Verlet (kick–drift–kick) integration, in place.

    phi : field, pi : conjugate momentum phi_t.  Both boundary values of pi
    must be 0 on entry (we enforce it), so the ends of phi never move.
    `on_sample(step, phi, pi)` is called every `sample_every` steps
    (and NOT at step 0 — sample the initial state yourself before calling).
    """
    pi[0] = pi[-1] = 0.0
    a = accel(phi, dx)
    for n in range(1, n_steps + 1):
        pi += 0.5 * dt * a          # half kick
        phi += dt * pi              # full drift
        a = accel(phi, dx)          # new force
        pi += 0.5 * dt * a          # half kick
        if n % sample_every == 0:
            on_sample(n, phi, pi)
    return phi, pi

# ----------------------------------------------------------------------
# 2. Diagnostics — everything is measured from the raw fields
# ----------------------------------------------------------------------

def grad_x(phi, dx):
    """Centered first difference (one-sided at the ends, where the field
    sits in vacuum and the gradient is ~0 anyway)."""
    g = np.empty_like(phi)
    g[1:-1] = (phi[2:] - phi[:-2]) / (2.0 * dx)
    g[0] = (phi[1] - phi[0]) / dx
    g[-1] = (phi[-1] - phi[-2]) / dx
    return g


def integrate(f, dx):
    """Trapezoid rule."""
    return float(dx * (f.sum() - 0.5 * (f[0] + f[-1])))


def energy_density(phi, pi, dx):
    """e = 1/2 phi_t^2 + 1/2 phi_x^2 + V(phi),  V = 1/2 (phi^2-1)^2."""
    gx = grad_x(phi, dx)
    return 0.5 * pi**2 + 0.5 * gx**2 + 0.25 * LAM * (phi**2 - VEV**2)**2


def total_energy(phi, pi, dx):
    return integrate(energy_density(phi, pi, dx), dx)


def total_momentum(phi, pi, dx):
    """Field momentum P = -integral phi_t phi_x dx (the T^{01} component of
    the stress tensor).  No particle mass appears anywhere in here."""
    return -integrate(pi * grad_x(phi, dx), dx)


def kink_center(x, phi):
    """Position of the kink = zero crossing of phi (upward, - to +),
    located by linear interpolation between the two straddling grid points."""
    idx = np.where((phi[:-1] < 0.0) & (phi[1:] >= 0.0))[0]
    i = int(idx[0])
    return float(x[i] - phi[i] * (x[i + 1] - x[i]) / (phi[i + 1] - phi[i]))


def all_zero_crossings(x, phi):
    """All sign changes of phi (either direction), linearly interpolated.
    Used to track the kink AND antikink in the collision run."""
    s = np.sign(phi)
    idx = np.where(s[:-1] * s[1:] < 0.0)[0]
    return [float(x[i] - phi[i] * (x[i + 1] - x[i]) / (phi[i + 1] - phi[i]))
            for i in idx]


def fit_tanh_profile(x, phi, center_guess, width_guess, half_window=10.0):
    """Least-squares fit of tanh((x-a)/b) in a window of +-10 around the
    kink.  Returns (a, b) = fitted center and width."""
    m = np.abs(x - center_guess) <= half_window
    popt, _ = curve_fit(lambda xx, a, b: np.tanh((xx - a) / b),
                        x[m], phi[m], p0=(center_guess, width_guess))
    return float(popt[0]), float(popt[1])

# ----------------------------------------------------------------------
# 3. Initial data
# ----------------------------------------------------------------------

def boosted_kink(x, u, x0):
    """Exact boosted kink at t = 0 (see module docstring for the signs)."""
    g = 1.0 / np.sqrt(1.0 - u**2)
    arg = g * (x - x0)
    phi = np.tanh(arg)
    # sech^2 = 1/cosh^2; clip the argument so cosh^2 cannot overflow float64
    # (sech^2(350) is already exactly 0.0 in double precision)
    pi = -u * g / np.cosh(np.clip(arg, -350.0, 350.0))**2
    return phi, pi


def sech2(arg):
    """Overflow-safe sech^2."""
    return 1.0 / np.cosh(np.clip(arg, -350.0, 350.0))**2

# ----------------------------------------------------------------------
# 4. Experiment A + B: velocity sweep with conservation tracking
# ----------------------------------------------------------------------
print("phi^4 kink simulation  (lambda = 2, v = 1, c = 1;  M = 4/3)")
print("=" * 100)

DX = 0.025
DT = 0.01
X = np.arange(-30.0, 130.0 + 0.5 * DX, DX)     # 6401 points
T_END = 80.0
N_STEPS = int(round(T_END / DT))               # 8000
SAMPLE_EVERY = 50                              # sample every dt*50 = 0.5

U_LIST = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
SNAP_TIMES = [0.0, 20.0, 40.0, 60.0, 80.0]     # energy-density snapshots (u=0.8)

sweep = {"u": [], "E": [], "P": [], "gammaM": [], "gammaMu": [],
         "width": [], "invGamma": [], "vFit": [], "driftMax": []}
drift_curve = None            # (t, relE) for u = 0.8
snapshots_full = None         # full-resolution e(x) at SNAP_TIMES for u = 0.8
centers_08 = None             # center trajectory for u = 0.8 (for the record)

for u in U_LIST:
    gamma = 1.0 / np.sqrt(1.0 - u**2)
    phi, pi = boosted_kink(X, u, x0=0.0)
    E0 = total_energy(phi, pi, DX)

    # per-run sample buffers
    t_samples = [0.0]
    E_samples = [E0]
    c_samples = [kink_center(X, phi)]
    snaps = {0.0: energy_density(phi, pi, DX).copy()} if u == 0.8 else {}

    def on_sample(n, phi, pi, _t=t_samples, _E=E_samples, _c=c_samples,
                  _s=snaps, _u=u):
        t = n * DT
        _t.append(t)
        _E.append(total_energy(phi, pi, DX))
        _c.append(kink_center(X, phi))
        if _u == 0.8 and any(abs(t - ts) < 1e-9 for ts in SNAP_TIMES):
            _s[t] = energy_density(phi, pi, DX).copy()

    evolve(phi, pi, DX, DT, N_STEPS, SAMPLE_EVERY, on_sample)

    t_arr = np.array(t_samples)
    E_arr = np.array(E_samples)
    c_arr = np.array(c_samples)

    # --- measurements -------------------------------------------------
    E_final = total_energy(phi, pi, DX)
    P_final = total_momentum(phi, pi, DX)
    drift_max = float(np.max(np.abs(E_arr - E0)) / E0)

    # speed: straight-line fit of the center trajectory, t >= 2 to skip
    # any (tiny) start-up transient of the discretized initial data
    mfit = t_arr >= 2.0
    vfit = float(np.polyfit(t_arr[mfit], c_arr[mfit], 1)[0])

    # width: tanh fit around the final center (lump is >= 20 units from
    # every boundary for all u in the sweep — checked below)
    c_now = c_arr[-1]
    assert c_now - X[0] > 20.0 and X[-1] - c_now > 20.0, \
        "measurement too close to a boundary"
    _, b_fit = fit_tanh_profile(X, phi, c_now, 1.0 / gamma)

    sweep["u"].append(u)
    sweep["E"].append(E_final)
    sweep["P"].append(P_final)
    sweep["gammaM"].append(gamma * M_KINK)
    sweep["gammaMu"].append(gamma * M_KINK * u)
    sweep["width"].append(b_fit)
    sweep["invGamma"].append(1.0 / gamma)
    sweep["vFit"].append(vfit)
    sweep["driftMax"].append(drift_max)

    if u == 0.8:
        drift_curve = (t_arr, (E_arr - E0) / E0)
        snapshots_full = snaps
        centers_08 = (t_arr, c_arr)

# ----------------------------------------------------------------------
# 5. Experiment C: stability of the kink against a hard poke
# ----------------------------------------------------------------------
# Static kink + Gaussian bump of amplitude 0.08 centered at x = 5 (five
# widths from the kink core).  If the kink were a fragile balance it would
# fall apart; being a topological soliton, it must relax back, radiating
# the excess energy away as small massive waves.
#
# Dedicated grid: the left wall sits at x = -45 so that even the fastest
# (near-luminal) radiation launched from the poke at x = 5 cannot reflect
# and re-enter the width-fit window |x| <= 10 before t_final = 60
# (wall hit at t = 50, earliest re-entry at t = 85); the right wall at
# x = 80 is 75 units away and is not reached at all.  So every measurement
# below is strictly reflection-free.
XS = np.arange(-45.0, 80.0 + 0.5 * DX, DX)     # 5001 points
phi_s = np.tanh(XS) + 0.08 * np.exp(-((XS - 5.0) ** 2) / 2.0)
pi_s = np.zeros_like(XS)
T_STAB = 60.0
N_STAB = int(round(T_STAB / DT))

phi_s0 = phi_s.copy()
e_s0 = energy_density(phi_s, pi_s, DX)
E_s0 = total_energy(phi_s, pi_s, DX)
stab_E = [E_s0]

def on_sample_stab(n, phi, pi):
    stab_E.append(total_energy(phi, pi, DX))

evolve(phi_s, pi_s, DX, DT, N_STAB, SAMPLE_EVERY, on_sample_stab)
e_sT = energy_density(phi_s, pi_s, DX)

stab_center, stab_width = fit_tanh_profile(
    XS, phi_s, kink_center(XS, phi_s), 1.0)
stab_drift = float(np.max(np.abs(np.array(stab_E) - E_s0)) / E_s0)
stab_ok = (abs(stab_width - 1.0) < 0.05) and (abs(stab_center) < 1.0)

# ----------------------------------------------------------------------
# 6. Experiment D: kink–antikink collision at u = +-0.5
# ----------------------------------------------------------------------
# Own grid, symmetric about 0.  Kink at -15 moving right, antikink at +15
# moving left.  u = 0.5 > v_c ~ 0.26, so phi^4 lore says: one bounce, then
# re-separation (inelastic — some energy is radiated and the pair exits
# slower than 0.5).
XC = np.arange(-60.0, 60.0 + 0.5 * DX, DX)     # 4801 points
U_COLL = 0.5
A_SEP = 15.0
G_COLL = 1.0 / np.sqrt(1.0 - U_COLL**2)

phi_c = np.tanh(G_COLL * (XC + A_SEP)) - np.tanh(G_COLL * (XC - A_SEP)) - 1.0
pi_c = (-U_COLL * G_COLL * sech2(G_COLL * (XC + A_SEP))
        - U_COLL * G_COLL * sech2(G_COLL * (XC - A_SEP)))

T_COLL = 60.0
N_COLL = int(round(T_COLL / DT))
E_c0 = total_energy(phi_c, pi_c, DX)

# heatmap meshes: a fine one for the PNG, a coarse one for curves.json
FINE_XSTRIDE = 8                     # 601 x-points for the figure
FINE_TSTEP = 30                      # every 0.3 -> 201 t-samples
coll_fine_t, coll_fine_e = [0.0], [energy_density(phi_c, pi_c, DX)[::FINE_XSTRIDE].copy()]
coll_E = [E_c0]
coll_seps = [(0.0, all_zero_crossings(XC, phi_c))]

def on_sample_coll(n, phi, pi):
    t = n * DT
    if n % FINE_TSTEP == 0:
        coll_fine_t.append(t)
        coll_fine_e.append(energy_density(phi, pi, DX)[::FINE_XSTRIDE].copy())
    coll_E.append(total_energy(phi, pi, DX))
    coll_seps.append((t, all_zero_crossings(XC, phi)))

evolve(phi_c, pi_c, DX, DT, N_COLL, 10, on_sample_coll)   # sample every 0.1

coll_drift = float(np.max(np.abs(np.array(coll_E) - E_c0)) / E_c0)

def pair_separation(crossings):
    return (max(crossings) - min(crossings)) if len(crossings) >= 2 else 0.0

sep_of_t = {round(t, 3): pair_separation(c) for t, c in coll_seps}
sep40, sep50, sep60 = sep_of_t[40.0], sep_of_t[50.0], sep_of_t[60.0]
# re-separation = the lumps are far apart again at the end and still receding
coll_reseparated = (sep60 > 15.0) and (sep60 > sep50 > sep40)
final_cross = coll_seps[-1][1]
coll_ok = coll_reseparated and (coll_drift < 1e-3)

# ----------------------------------------------------------------------
# 7. Acceptance checks
# ----------------------------------------------------------------------
rows = []
all_pass = True
for k, u in enumerate(sweep["u"]):
    E, P = sweep["E"][k], sweep["P"][k]
    gM, gMu = sweep["gammaM"][k], sweep["gammaMu"][k]
    b, ig = sweep["width"][k], sweep["invGamma"][k]
    vf, dr = sweep["vFit"][k], sweep["driftMax"][k]

    tol_E = 0.01 if u >= 0.95 else 0.005
    tol_P = 0.01 if u >= 0.95 else 0.005
    tol_w = 0.03 if u >= 0.95 else 0.02

    e_ratio = E / gM
    p_ratio = (P / gMu) if u > 0 else None      # gMu = 0 at u = 0
    ok_E = abs(e_ratio - 1.0) < tol_E
    ok_P = (abs(p_ratio - 1.0) < tol_P) if u > 0 else (abs(P) < 1e-6)
    ok_v = abs(vf - u) < 0.005
    ok_w = abs(b / ig - 1.0) < tol_w
    ok_d = dr < 1e-4
    row_pass = ok_E and ok_P and ok_v and ok_w and ok_d
    all_pass &= row_pass
    rows.append(dict(u=u, E=E, gammaM=gM, E_ratio=e_ratio, P=P, gammaMu=gMu,
                     P_ratio=p_ratio, vFit=vf, width=b, invGamma=ig,
                     driftMax=dr, ok_E=ok_E, ok_P=ok_P, ok_v=ok_v, ok_w=ok_w,
                     ok_drift=ok_d, passed=row_pass))

all_pass &= stab_ok and coll_ok

# ----------------------------------------------------------------------
# 8. JSON outputs
# ----------------------------------------------------------------------

def sig5(obj):
    """Round every float in a nested structure to 5 significant digits."""
    if isinstance(obj, dict):
        return {k: sig5(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sig5(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        return float(f"{float(obj):.5g}")
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj

# ---- results.json: every measured number + pass/fail bookkeeping -------
results = {
    "model": {"lambda": LAM, "v": VEV, "M_analytic": M_KINK,
              "potential": "V(phi) = (lambda/4)(phi^2 - v^2)^2",
              "eom": "phi_tt = phi_xx - lambda*phi*(phi^2 - v^2)"},
    "numerics": {"dx": DX, "dt": DT, "x_range": [float(X[0]), float(X[-1])],
                 "t_end_sweep": T_END, "scheme": "velocity-Verlet leapfrog",
                 "boundaries": "Dirichlet (ends pinned)"},
    "tolerances": {"E_ratio": "0.5% (1% at u=0.95)",
                   "P_ratio": "0.5% (1% at u=0.95)",
                   "vFit": "|vFit-u| < 0.005",
                   "width": "2% of 1/gamma (3% at u=0.95)",
                   "energy_drift": "< 1e-4 relative (sweep), < 1e-3 (collision)"},
    "sweep": rows,
    "stability": {"perturbation": "0.08*exp(-(x-5)^2/2) added to phi at t=0",
                  "grid": [float(XS[0]), float(XS[-1])],
                  "reflection_free": "left-wall round trip to fit window = 85 > t_final",
                  "t_final": T_STAB, "fitted_center": stab_center,
                  "fitted_width": stab_width, "energy_driftMax": stab_drift,
                  "criteria": "width within 5% of 1, center within 1 of origin",
                  "passed": bool(stab_ok)},
    "collision": {"u": U_COLL, "initial_separation": 2 * A_SEP,
                  "grid": [-60.0, 60.0], "t_final": T_COLL,
                  "energy_driftMax": coll_drift,
                  "separation_t40": sep40, "separation_t50": sep50,
                  "separation_t60": sep60,
                  "final_crossings": final_cross,
                  "reseparated": bool(coll_reseparated),
                  "passed": bool(coll_ok)},
    "all_pass": bool(all_pass),
    "runtime_seconds": None,   # filled just before writing
}

# ---- curves.json: the exact shared schema for the HTML page ------------
# snapshots (u = 0.8): x window [-15, 85], stride 11 -> 364 points
snap_mask = (X >= -15.0) & (X <= 85.0)
snap_x = X[snap_mask][::11]
snap_e = [snapshots_full[t][snap_mask][::11] for t in SNAP_TIMES]

# drift (u = 0.8): thin the 161 half-unit samples to 120
t_dr, rel_dr = drift_curve
pick = np.unique(np.round(np.linspace(0, len(t_dr) - 1, 120)).astype(int))
drift_json = {"t": t_dr[pick].tolist(), "relE": rel_dr[pick].tolist()}

# collision heatmap: coarse mesh <=160 x-points x <=100 t-samples,
# thinned from the fine mesh recorded during the run
fine_e = np.array(coll_fine_e)                 # (201, 601) on XC[::8]
fine_x = XC[::FINE_XSTRIDE]
cx_mask = (fine_x >= -45.0) & (fine_x <= 45.0)  # 451 fine x-points
cx = fine_x[cx_mask][::3]                       # 151 x-points
ct_pick = np.unique(np.round(np.linspace(0, len(coll_fine_t) - 1, 100)).astype(int))
ce = [fine_e[i][cx_mask][::3].tolist() for i in ct_pick]

# stability profiles: x window [-30, 60], stride 10 -> 361 points
# (half-dx tolerance so float rounding in arange cannot drop an endpoint)
stab_mask = (XS >= -30.0 - 0.5 * DX) & (XS <= 60.0 + 0.5 * DX)
stab_x = XS[stab_mask][::10]

curves = {
    "M": 1.33333,
    "sweep": {k: list(v) for k, v in sweep.items()},
    "drift": drift_json,
    "snapshots": {"x": snap_x.tolist(), "t": list(SNAP_TIMES),
                  "e": [e.tolist() for e in snap_e]},
    "collision": {"x": cx.tolist(),
                  "t": [coll_fine_t[i] for i in ct_pick],
                  "e": ce},
    "stability": {"x": stab_x.tolist(),
                  "phi0": phi_s0[stab_mask][::10].tolist(),
                  "phiT": phi_s[stab_mask][::10].tolist(),
                  "e0": e_s0[stab_mask][::10].tolist(),
                  "eT": e_sT[stab_mask][::10].tolist(),
                  "tFinal": 60},
}

results["runtime_seconds"] = time.time() - T_START
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(sig5(results), f, indent=1)
curves_rounded = sig5(curves)
curves_rounded["M"] = 1.33333          # keep the 6-digit value from the contract
with open(os.path.join(OUT, "curves.json"), "w") as f:
    json.dump(curves_rounded, f, separators=(",", ":"))

# ----------------------------------------------------------------------
# 9. Figures
# ----------------------------------------------------------------------
SURFACE, INK, SECOND = "#fcfcfb", "#0b0b0b", "#52514e"
HAIR, BLUE, AMBER, RED = "#e1e0d9", "#2a78d6", "#eda100", "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": SECOND,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": SECOND, "ytick.color": SECOND,
    "axes.grid": True, "grid.color": HAIR, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10.5, "axes.titlesize": 11.5, "legend.frameon": False,
})
PT = dict(color=AMBER, marker="o", markersize=8, linestyle="none",
          markeredgecolor=INK, markeredgewidth=0.6, zorder=5)

# --- fig 1: energy-density snapshots, u = 0.8 ---------------------------
fig, ax = plt.subplots(figsize=(8.4, 4.2))
shades = plt.cm.Blues(np.linspace(0.35, 0.95, len(SNAP_TIMES)))
for t, col in zip(SNAP_TIMES, shades):
    e = snapshots_full[t]
    ax.plot(X, e, color=col, lw=1.6)
    ipk = int(np.argmax(e))
    ax.annotate(f"t = {t:g}", (X[ipk], e[ipk]), xytext=(0, 6),
                textcoords="offset points", ha="center",
                color=SECOND, fontsize=9)
ax.set_xlim(-12, 82)
ax.set_ylim(0, 4.4)
ax.set_xlabel("x  [w]")
ax.set_ylabel(r"energy density  [Mc$^2$/w]")
ax.set_title("A moving kink is a rigid lump of field energy   (u = 0.8, snapshots every 20 time units)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_snapshots.png"), dpi=180)
plt.close(fig)

# --- fig 2: E vs gamma*M  and  P vs gamma*M*u ---------------------------
uu = np.linspace(0, 0.96, 400)
gg = 1.0 / np.sqrt(1.0 - uu**2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.1))
ax1.plot(uu, gg * M_KINK, color=BLUE, lw=2, label=r"$\gamma M$  (special relativity)")
ax1.plot(sweep["u"], sweep["E"], **PT, label="measured field energy")
ax1.axhline(M_KINK, color=SECOND, lw=1, ls=":")
ax1.text(0.02, M_KINK + 0.07, "rest mass  M = 4/3", color=SECOND, fontsize=9)
ax1.set_xlabel("kink speed  u  [c]")
ax1.set_ylabel("E  [natural units, c = 1]")
ax1.set_title(r"Energy:  E = $\gamma M c^2$")
ax1.legend(loc="upper left", fontsize=9)
ax2.plot(uu, gg * M_KINK * uu, color=BLUE, lw=2, label=r"$\gamma M u$  (special relativity)")
ax2.plot(sweep["u"], sweep["P"], **PT, label="measured field momentum")
ax2.set_xlabel("kink speed  u  [c]")
ax2.set_ylabel("P  [natural units, c = 1]")
ax2.set_title(r"Momentum:  P = $\gamma M u$")
ax2.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_energy_momentum.png"), dpi=180)
plt.close(fig)

# --- fig 3: energy conservation, u = 0.8 --------------------------------
fig, ax = plt.subplots(figsize=(8.0, 3.6))
ax.plot(drift_curve[0], drift_curve[1] * 1e6, color=AMBER, lw=1.6)
ax.axhline(0, color=SECOND, lw=0.8, ls=":")
dmax = sweep["driftMax"][U_LIST.index(0.8)]
ax.set_xlabel("t  [w/c]")
ax.set_ylabel(r"(E(t) $-$ E(0)) / E(0)   [$\times 10^{-6}$]")
ax.set_title(f"Energy conservation, u = 0.8   (max |drift| = {dmax:.2e}, target < 1e-4)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_conservation.png"), dpi=180)
plt.close(fig)

# --- fig 4: Lorentz contraction of the fitted width ---------------------
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.plot(uu, 1.0 / gg, color=BLUE, lw=2, label=r"$w/\gamma$  (Lorentz contraction)")
ax.plot(sweep["u"], sweep["width"], **PT, label="fitted width  b")
ax.set_xlabel("kink speed  u  [c]")
ax.set_ylabel("kink width  [w]")
ax.set_ylim(0, 1.1)
ax.set_title(r"The moving kink contracts:  b = w/$\gamma$")
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_contraction.png"), dpi=180)
plt.close(fig)

# --- fig 5: stability under a poke ---------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 5.6), sharex=True)
ax1.plot(XS, np.tanh(XS), color=BLUE, lw=2, ls="--", label="ideal kink  tanh(x)")
ax1.plot(XS, phi_s0, color=RED, lw=1.4, label="t = 0  (kink + poke)")
ax1.plot(XS, phi_s, color=INK, lw=1.4, label=f"t = {T_STAB:g}  (relaxed)")
ax1.set_ylabel(r"$\varphi$")
ax1.set_xlim(-30, 60)
ax1.legend(loc="lower right", fontsize=9)
ax1.set_title("Poke test: Gaussian pulse (amplitude 0.08 at x = 5) does not destroy the kink")
# log scale: the kink peak (e = 1) and the weak radiation (e ~ 1e-4)
# differ by four orders of magnitude, so a linear axis would hide the
# radiation entirely
ax2.semilogy(XS, np.maximum(e_s0, 1e-10), color=RED, lw=1.4,
             label="energy density, t = 0")
ax2.semilogy(XS, np.maximum(e_sT, 1e-10), color=INK, lw=1.4,
             label=f"energy density, t = {T_STAB:g}")
ax2.annotate("poke", xy=(5, 0.02), color=RED, fontsize=9, ha="center")
ax2.annotate("radiation dispersed over the box\n(leaving the kink untouched)",
             xy=(-18, 3e-3), color=SECOND, fontsize=9, ha="center")
ax2.set_xlabel("x  [w]")
ax2.set_ylabel(r"e(x)  [Mc$^2$/w],  log scale")
ax2.set_ylim(1e-8, 8)
ax2.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_stability.png"), dpi=180)
plt.close(fig)

# --- fig 6: kink–antikink collision heatmap ------------------------------
fig, ax = plt.subplots(figsize=(7.6, 5.4))
im = ax.imshow(fine_e, origin="lower", aspect="auto", cmap="Blues",
               extent=[float(fine_x[0]), float(fine_x[-1]),
                       0.0, float(coll_fine_t[-1])],
               vmin=0.0, vmax=float(np.percentile(fine_e, 99.7)))
ax.set_xlim(-45, 45)
ax.set_xlabel("x  [w]")
ax.set_ylabel("t  [w/c]")
ax.set_title(f"Kink–antikink collision at u = ±0.5 > $v_c$ ≈ 0.26:  bounce and re-separation\n"
             f"(energy density;  max drift = {coll_drift:.1e})")
ax.grid(False)
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label(r"e(x,t)  [Mc$^2$/w]", color=INK)
cb.ax.yaxis.set_tick_params(color=SECOND)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_collision.png"), dpi=180)
plt.close(fig)

# ----------------------------------------------------------------------
# 10. Summary table
# ----------------------------------------------------------------------
hdr = (f"{'u':>5} {'E':>9} {'gammaM':>9} {'E/gM':>8} {'P':>9} {'gammaMu':>9} "
       f"{'P/gMu':>8} {'vFit':>8} {'width':>7} {'1/gamma':>8} {'driftMax':>9} {'pass':>5}")
print(hdr)
print("-" * len(hdr))
for r in rows:
    pr = f"{r['P_ratio']:.5f}" if r["P_ratio"] is not None else "   --  "
    print(f"{r['u']:>5.2f} {r['E']:>9.5f} {r['gammaM']:>9.5f} {r['E_ratio']:>8.5f} "
          f"{r['P']:>9.5f} {r['gammaMu']:>9.5f} {pr:>8} {r['vFit']:>8.5f} "
          f"{r['width']:>7.4f} {r['invGamma']:>8.4f} {r['driftMax']:>9.2e} "
          f"{'OK' if r['passed'] else 'FAIL':>5}")
print("-" * len(hdr))
print(f"stability : fitted width = {stab_width:.4f} (target 1 +- 5%), "
      f"center = {stab_center:+.4f} (target |c| < 1), drift = {stab_drift:.2e}"
      f"  -> {'OK' if stab_ok else 'FAIL'}")
print(f"collision : separation t=40/50/60 = {sep40:.2f}/{sep50:.2f}/{sep60:.2f}, "
      f"re-separated = {coll_reseparated}, drift = {coll_drift:.2e}"
      f"  -> {'OK' if coll_ok else 'FAIL'}")
print(f"ALL TARGETS {'PASS' if all_pass else 'FAIL'}   "
      f"(wall clock: {time.time() - T_START:.1f} s)")
