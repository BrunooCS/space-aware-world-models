"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Shared resumable dataset generation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1) Scene maps: dispatch missing seed batches to environment-specific workers.
   Existing PNGs are treated as complete.
2) Rollouts: workers write one HDF5 shard per episode, then the shards are merged
   into a single resumable dataset. Completed shards and merged episodes are skipped.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import multiprocessing as mp
import shutil
import time
from functools import partial
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# ── Transform / Core Logic ───────────────────────────
def generate_dataset_parallel(cfg, worker):
    """Generate cfg.num_samples distinct seed-named PNGs from cfg.seed, resuming existing files."""
    # ── Prepare seeds and output ──────────────────────
    start = time.monotonic()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(cfg.seed)
    seeds = dict.fromkeys(map(int, rng.randint(0, 2**31 - 1, size=cfg.num_samples)))
    # Replace duplicate seeds without reordering unique draws.
    while len(seeds) < cfg.num_samples:
        seeds.update(dict.fromkeys(map(int, rng.randint(0, 2**31 - 1, size=cfg.num_samples - len(seeds)))))
    missing = [seed for seed in seeds if not (output_dir / f"{seed}.png").exists()]
    if not missing:
        print(f"[INFO] All {cfg.num_samples} PNGs exist in {output_dir}")
        return

    # ── Generate missing batches ──────────────────────
    batches = [missing[i:i + cfg.chunk_size] for i in range(0, len(missing), cfg.chunk_size)]
    generate = partial(worker, img_size=(cfg.img_width, cfg.img_height), output_dir=output_dir)
    print(f"[INFO] Generating {len(missing)}/{cfg.num_samples} PNGs with {cfg.num_processes} workers")
    with mp.Pool(cfg.num_processes) as pool:
        completed = sum(tqdm(pool.imap(generate, batches), total=len(batches), desc="Batches"))

    # ── Report output ─────────────────────────────────
    elapsed = time.monotonic() - start
    print(f"[INFO] Saved {completed} PNGs in {elapsed:.1f}s → {output_dir}")


# ── Rollout shards ───────────────────────────────────
_PROGRESS_COUNTER = None


def _init_progress_counter(counter):
    """Share the parent process's episode counter with pool workers."""
    global _PROGRESS_COUNTER
    _PROGRESS_COUNTER = counter


def _advance_progress():
    with _PROGRESS_COUNTER.get_lock():
        _PROGRESS_COUNTER.value += 1


def shard_path(tmp_dir, episode_id):
    return Path(tmp_dir) / f"episode_{episode_id:06d}.h5"


def worker_loop(worker_id, episode_ids, cfg, completed_episodes, make_env, rollout):
    """Record the missing shards of `episode_ids` in one environment; returns (worker_id, shards written, seconds)."""
    start, generated = time.monotonic(), 0
    env = make_env(cfg)
    try:
        for episode_id in episode_ids:
            if episode_id in completed_episodes:
                continue
            path = shard_path(cfg.tmp_dir, episode_id)
            if path.exists():
                try:
                    with h5py.File(path, "r") as f:
                        done = f.attrs.get("completed", False)
                except OSError:
                    done = False  # shard truncated by an interrupted run: regenerate it
                if done:
                    _advance_progress()
                    continue
                path.unlink()
            rollout(env, path, episode_id, cfg)
            generated += 1
            _advance_progress()
    finally:
        env.close()
    return worker_id, generated, time.monotonic() - start


def merge_episodes(cfg, attrs):
    """Merge completed shards into `cfg.output_h5`; the merged datasets take their dtype, compression, and per-episode chunks from the shards."""
    output_path = Path(cfg.output_h5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "a") as output:
        if "episodes" not in output:
            with h5py.File(shard_path(cfg.tmp_dir, 0), "r") as shard:
                episodes = output.create_group("episodes")
                for name, data in shard.items():
                    if name != "seed":
                        chunks = None if data.chunks is None else (1, *data.chunks)
                        episodes.create_dataset(name, (cfg.num_episodes, *data.shape), data.dtype, chunks=chunks,
                                                compression=data.compression, compression_opts=data.compression_opts)
            output.create_dataset("seeds", (cfg.num_episodes,), np.int64)
            output.create_dataset("written_mask", (cfg.num_episodes,), np.bool_)
            output.attrs.update({**attrs, "num_episodes": cfg.num_episodes})
        episodes, seeds, written = output["episodes"], output["seeds"], output["written_mask"]

        merged = 0
        for episode_id in np.flatnonzero(~written[...]):
            path = shard_path(cfg.tmp_dir, episode_id)
            with h5py.File(path, "r") as shard:
                for name, data in episodes.items():
                    data[episode_id] = shard[name][...]
                seeds[episode_id] = shard["seed"][()]
            written[episode_id] = True
            path.unlink()
            merged += 1
        print(f"[INFO] Merged {merged} new episodes; {int(np.count_nonzero(written[...]))}/{cfg.num_episodes} complete")
    shutil.rmtree(cfg.tmp_dir)


def generate_rollouts(cfg, make_env, rollout, attrs):
    """Generate the missing episodes of `cfg.num_episodes` in `cfg.num_workers` processes and merge them into `cfg.output_h5`.

    `make_env(cfg)` builds one environment per worker, `rollout(env, shard_path, episode_id, cfg)` records one shard, and
    `attrs` are the file-level attributes of a new dataset.
    """
    output_path = Path(cfg.output_h5)
    Path(cfg.tmp_dir).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Episodes: {cfg.num_episodes} | Steps: {cfg.steps_per_episode} | Policy: {cfg.policy}")
    print(f"[INFO] Temp dir: {cfg.tmp_dir} | Output: {cfg.output_h5}")

    completed_episodes = set()
    if output_path.exists():
        with h5py.File(output_path, "r") as output:
            written_mask = output["written_mask"][...]
            if written_mask.shape[0] != cfg.num_episodes:
                raise ValueError(f"Existing dataset episode count does not match {cfg.num_episodes}")
            completed_episodes = set(np.flatnonzero(written_mask).tolist())
    print(f"[INFO] Found {len(completed_episodes)} merged episodes")

    # Round-robin episode assignment
    assignments = [range(worker_id, cfg.num_episodes, cfg.num_workers) for worker_id in range(cfg.num_workers)]
    tasks = [(worker_id, episode_ids, cfg, completed_episodes, make_env, rollout) for worker_id, episode_ids in enumerate(assignments) if episode_ids]
    if tasks:
        counter = mp.Value("i", len(completed_episodes))
        with mp.Pool(cfg.num_workers, initializer=_init_progress_counter, initargs=(counter,)) as pool:
            result = pool.starmap_async(worker_loop, tasks)
            with tqdm(total=cfg.num_episodes, initial=len(completed_episodes), desc="Episodes", unit="episode") as progress:
                while not result.ready():
                    progress.update(counter.value - progress.n)
                    time.sleep(0.1)
                worker_results = result.get()
                progress.update(counter.value - progress.n)
        for worker_id, generated, elapsed in worker_results:
            print(f"[INFO] Worker {worker_id}: generated {generated} episodes in {elapsed:.1f}s")

    merge_episodes(cfg, attrs)
    print(f"[INFO] Dataset complete: {output_path}")
