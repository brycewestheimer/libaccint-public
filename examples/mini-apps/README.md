# LibAccInt Mini-Applications

Mini-apps showing argument-driven RHF workflows built around LibAccInt's
ShellSet work-unit model:

- **1e integrals**: iterate `BasisSet.shell_set_pairs()`
- **2e integrals**: iterate `BasisSet.shell_set_quartets()`
- **generic dispatch**: use `engine.compute(...)` with backend hints

Both mini-apps support:

- molecule and basis arguments
- CPU/GPU dispatch hints with CPU fallback
- two-electron modes:
  - `buffer`: compute quartets to `IntegralBuffer` and consume on host
  - `consumer`: compute quartets directly into `FockBuilder`
  - `compare`: run both and report differences

## Mini-Apps

- `cpp-hf/`: C++20 RHF mini-app
- `python-hf/`: Python RHF mini-app

See each subdirectory `README.md` for build/run details.
