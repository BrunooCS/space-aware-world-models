"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generate DMLab scene-map PNGs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1) Draw distinct seeds from the dataset seed and skip existing seed-named PNGs.
2) Render RGB maze maps in parallel, creating one environment per seed.

Requires an importable deepmind_lab installation. Custom builds can expose
bazel-bin through PYTHONPATH and pass --runfiles_path explicitly.
Run from the repository root:
    python -m src.datasets.make_topdown_views_dmlab
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import gc
from functools import partial

import pyrallis
from PIL import Image

from src.configs.datasets import DMLabMapDatasetConfig
from src.datasets.generation import generate_dataset_parallel
from src.envs.dm_lab_maze import make_random_maze_env

# ── Transform / Core Logic ───────────────────────────
def generate_sample_batch(batch_seeds, img_size, output_dir, runfiles_path=None):
    """Save one RGB scene map m per seed and return the completed count."""
    width, height = img_size
    for seed in batch_seeds:
        env = make_random_maze_env(width=width, height=height, topdown_width=width,
                                   topdown_height=height, runfiles_path=runfiles_path)
        try:
            _, _, info = env.reset(seed=int(seed))
            scene_map = info["topdown_rgb"]  # m: [H, W, 3], uint8 RGB
            Image.fromarray(scene_map).save(output_dir / f"{seed}.png")
        finally:
            env.close()
        del env, scene_map, info
        gc.collect()  # release native environment resources between seeds
    return len(batch_seeds)

# ── Main ─────────────────────────────────────────────
def main():
    """Parse dataset settings and generate the missing maze maps."""
    cfg = pyrallis.parse(config_class=DMLabMapDatasetConfig)
    worker = partial(generate_sample_batch, runfiles_path=cfg.runfiles_path)
    generate_dataset_parallel(cfg, worker)

if __name__ == "__main__":
    main()
