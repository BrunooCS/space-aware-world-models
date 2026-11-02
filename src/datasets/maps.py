"""Top-down scene-map datasets decoded once from seed-named RGB PNG folders into a shared uint8 tensor."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch.utils.data import BatchSampler, Dataset, RandomSampler, SequentialSampler
from torchvision.io import ImageReadMode, read_image


# ── Inputs / Data ────────────────────────────────────
class TopDownMapDataset(Dataset):
    """Scene maps m as uint8 [3, H, W] gathered from an in-memory tensor; a list of indices returns a batch [B, 3, H, W]."""

    def __init__(self, maps, indices, flip_augment=False):
        self.maps, self.indices, self.flip_augment = maps, indices, flip_augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        m = self.maps[self.indices[index]]
        if self.flip_augment:  # independent horizontal and vertical flips per map of the batch
            m = torch.where(torch.rand(len(m), 1, 1, 1) < 0.5, m.flip(-1), m)
            m = torch.where(torch.rand(len(m), 1, 1, 1) < 0.5, m.flip(-2), m)
        return m


def load_maps(map_dir):
    """Decode every PNG of `map_dir` (sorted by name) into one uint8 tensor [N, 3, H, W]."""
    paths = sorted(Path(map_dir).glob("*.png"))
    with ThreadPoolExecutor() as pool:  # PNG decoding releases the GIL
        return torch.stack(list(pool.map(lambda path: read_image(str(path), mode=ImageReadMode.RGB), paths)))


def split_maps(map_dir, val_fraction, seed, flip_augment=False):
    """Seeded train/validation split over one shared tensor; only the training subset is augmented."""
    maps = load_maps(map_dir)
    indices = torch.randperm(len(maps), generator=torch.Generator().manual_seed(seed))
    num_val = int(len(indices) * val_fraction)
    return TopDownMapDataset(maps, indices[num_val:], flip_augment), TopDownMapDataset(maps, indices[:num_val])


def batch_sampler(dataset, batch_size, train):
    """Batch-index sampler: pass as `DataLoader(dataset, sampler=..., batch_size=None)` to gather whole batches from memory."""
    sampler = RandomSampler(dataset) if train else SequentialSampler(dataset)
    return BatchSampler(sampler, batch_size, drop_last=train)
