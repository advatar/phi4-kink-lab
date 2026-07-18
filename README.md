# The φ⁴ Kink Lab

**Live page: https://advatar.github.io/phi4-kink-lab/**

An interactive toy model of one idea: *mass is trapped field energy*. The page runs a real
1+1-dimensional φ⁴ field-theory integrator live in your browser. The field equation

φ_tt = φ_xx − 2φ(φ² − 1)

contains **no mass parameter** — yet its kink soliton carries a rest mass M = 4/3 (computed,
not postulated), obeys E = γMc² and P = γMu under boosts, Lorentz-contracts as 1/γ, survives
perturbations, and collides like a particle. All of that is *measured off the screen*, and
independently re-verified offline at higher resolution by `kink_sim.py`.

The page is a single self-contained HTML file: no dependencies, no network requests, works
offline.

## Contents

| File | What it is |
|---|---|
| `index.html` | The lab — everything inlined, including the verification data and the Python source |
| `kink_sim.py` | The standalone simulation that produced the verification numbers |
| `out/` | Its outputs: figures and measured results (`results.json`) |
| [`CITU_PRD_manuscript_v2.pdf`](CITU_PRD_manuscript_v2.pdf) | The companion manuscript (G. Kranck): confined field energy as inertia, and an exact Z₃ identity underlying the Koide charged-lepton mass relation |

## Running the simulation yourself

```
python3 -m venv venv && ./venv/bin/pip install numpy scipy matplotlib
./venv/bin/python kink_sim.py
```

Runs in a few seconds; prints a summary table and regenerates `out/`.

Key verified numbers (leapfrog, dx = 0.025, dt = 0.01, u = 0 → 0.95): E matches γMc² to
better than 0.1%, momentum to 0.1%, fitted width matches 1/γ to 0.03%, energy drift ≤ 1.4×10⁻⁵,
and a kink–antikink pair at u = 0.5 bounces — particle phenomenology with no particles in the
equations.

## Honest scope

The kink demonstrates the *mechanism* (mass from confined field energy, relativity for free).
It does not establish existence in three dimensions (Derrick's theorem is the standing
obstruction), any specific mass spectrum, spin, gravity, or quantum behavior — the page's
closing ledger spells this out.
