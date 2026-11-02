"""Checkpoint loading and posterior-mean scene encoding for SWM viewers."""

import torch

from src.models.dds_vae import DDSVAE, normalize_maps
from src.models.generator import Generator
from src.models.transition import TransitionModel


# ── Model ────────────────────────────────────────────
def load_swm(checkpoint, generator, device):
    """Load the environment, scene encoder, renderer, and transition model from compatible checkpoints."""
    hparams = torch.load(checkpoint, map_location="cpu")["hyper_parameters"]
    renderer_encoder = torch.load(generator, map_location="cpu")["hyper_parameters"]["scene_encoder"]
    # assert renderer_encoder == hparams["scene_encoder"], f"renderer scene encoder {renderer_encoder} != {hparams['scene_encoder']}"
    return (hparams["env"], DDSVAE.from_checkpoint(hparams["scene_encoder"], device), Generator.from_checkpoint(generator, device),
            TransitionModel.from_checkpoint(checkpoint, device))


@torch.inference_mode()
def encode_scene(scene_encoder, scene_map, device):
    """Encode one uint8 RGB scene map [H, W, 3] as its posterior mean [1, d_z]."""
    # [H, W, 3] uint8 -> [1, 3, H, W] in [-1, 1].
    m = normalize_maps(torch.as_tensor(scene_map, device=device).permute(2, 0, 1)[None])
    return scene_encoder.encode(m)[0]
