# Setup Guide

This guide gets you from a fresh Linux machine to a working environment for
`lagrangian-mbrl-franka`. Target hardware: a single **NVIDIA RTX 4070 (12 GB)**.

> Summary of dependencies is in [`../README.md`](../README.md); this is the
> detailed walkthrough.

## 1. Prerequisites

- **OS:** Linux (Ubuntu 22.04 recommended; tested under WSL2 as well).
- **GPU:** NVIDIA RTX (≥ 8 GB; 12 GB on the 4070 is the project target).
- **Driver + CUDA:** a recent NVIDIA driver compatible with the Isaac Sim
  release you install. Verify with `nvidia-smi`.
- **Python:** 3.10 (match what Isaac Lab supports for your release).
- **Disk:** Isaac Sim is large (tens of GB); ensure ample free space.

## 2. Install Isaac Sim + Isaac Lab (2.x)

Isaac Sim/Isaac Lab have their own installer and are **not** pulled in by this
package. Follow the official guide and pick the **stable 2.x** line:

- Isaac Lab docs & install: <https://isaac-sim.github.io/IsaacLab/>

Recommended path (per the official docs):
1. Install Isaac Sim (binary or pip, per the current Isaac Lab instructions).
2. Clone Isaac Lab, run its `./isaaclab.sh --install` (or documented equivalent).
3. Use the conda/venv environment Isaac Lab sets up — install everything else
   into **that same** environment.

### Verify Isaac Lab works (headless)
Run one of Isaac Lab's bundled Franka tasks headless to confirm the GPU path:

```bash
# Example shape — use the exact task/runner from your Isaac Lab version's docs.
python -m isaaclab.envs ... --task Isaac-Reach-Franka-v0 --headless
```

If this launches and steps without error, the simulator side is good.

### VRAM tips for the 4070 (12 GB)
- Run **headless** (`--headless`).
- Keep `env.num_envs` modest (start at 64; scale up only if VRAM allows).
- Prefer fp32 dynamics but watch memory; reduce batch sizes if you hit OOM.

## 3. Install this package

Into the **same** Isaac Lab environment:

```bash
git clone <your-fork-url> lagrangian-mbrl-franka
cd lagrangian-mbrl-franka

# editable install (reads pyproject.toml)
pip install -e .

# with RL baselines and/or dev tooling:
pip install -e ".[rl,dev]"
```

## 4. Verify the install

```bash
python -c "import lagrangian_mbrl; print(lagrangian_mbrl.__version__)"
pytest -q            # smoke + util tests pass; DeLaN tests xfail until implemented
```

## 5. Experiment tracking (optional but recommended)

```bash
wandb login          # once, for cloud experiment tracking
# TensorBoard logs land under logs/ by default.
```

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `nvidia-smi` not found / no GPU | driver not installed | install NVIDIA driver; on WSL2 install the Windows-side driver. |
| Isaac Sim CUDA mismatch | driver/CUDA vs Sim release | match Sim release to your driver per Isaac Lab docs. |
| OOM at startup | too many envs | lower `env.num_envs`; run headless. |
| `import isaaclab` fails | wrong env active | activate the env Isaac Lab installed into. |

See also [`experiments_protocol.md`](experiments_protocol.md) for how to run and
reproduce experiments, and [`architecture.md`](architecture.md) for the code design.
