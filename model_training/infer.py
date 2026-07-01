import os
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm

from pillow_heif import register_heif_opener
register_heif_opener()

from models.unet    import UNet
from models.pix2pix import Pix2Pix


SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.heic', '.heif')

MODEL_REGISTRY = {
    'unet':    UNet,
    'pix2pix': Pix2Pix,
}

# Augmentation pairs (forward transform, inverse transform)
# Inverse = same as forward for all flip operations
TTA_AUGMENTATIONS = [
    (lambda x: x,                lambda x: x),                # original
    (lambda x: x.flip([3]),      lambda x: x.flip([3])),      # horizontal flip
    (lambda x: x.flip([2]),      lambda x: x.flip([2])),      # vertical flip
    (lambda x: x.flip([2, 3]),   lambda x: x.flip([2, 3])),   # both flips
]


# ── Device & cache helpers ────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def empty_cache(device: torch.device):
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()


# ── Tiled inference ───────────────────────────────────────────────────────────

def get_tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """Start positions that guarantee full coverage of [0, length]."""
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size, stride))
    starts.append(length - tile_size)   # ensure the far edge is always covered
    return sorted(set(starts))


def make_hann_mask(h: int, w: int) -> torch.Tensor:
    """
    2D Hann window weight mask for smooth tile blending.
    High in the centre, tapers to ~zero at the edges — prevents visible seams.
    Returns shape [1, 1, h, w].
    """
    wy = torch.hann_window(h, periodic=False)
    wx = torch.hann_window(w, periodic=False)
    return (wy[:, None] * wx[None, :])[None, None]   # [1, 1, H, W]


def run_tiled(model: torch.nn.Module, tensor: torch.Tensor,
              tile_size: int, overlap: int,
              device: torch.device) -> torch.Tensor:
    """
    Split a large image tensor into overlapping tiles, run the model on each,
    and blend back using a Hann window weight mask.

    - All tiles are exactly tile_size × tile_size (÷32 constraint satisfied).
    - Edge tiles are shifted inward rather than padded, so no border
      artefacts from padding are introduced.
    - Weight normalisation ensures correct blending in overlap regions.
    """
    _, c, H, W = tensor.shape
    stride = tile_size - overlap

    # If the image fits in a single tile, skip the overhead
    tile_h = min(tile_size, H)
    tile_w = min(tile_size, W)

    ys   = get_tile_starts(H, tile_h, stride)
    xs   = get_tile_starts(W, tile_w, stride)
    mask = make_hann_mask(tile_h, tile_w)

    accum  = torch.zeros(1, c, H, W)
    weight = torch.zeros(1, 1, H, W)

    total = len(ys) * len(xs)
    bar   = tqdm(total=total, desc="  tiles", leave=False, unit="tile")

    for y in ys:
        for x in xs:
            y2, x2 = min(y + tile_h, H), min(x + tile_w, W)
            y1, x1 = y2 - tile_h, x2 - tile_w  # shift inward at edges

            tile     = tensor[:, :, y1:y2, x1:x2].to(device)
            tile_out = model(tile).cpu()
            del tile

            accum [:, :, y1:y2, x1:x2] += tile_out * mask
            weight[:, :, y1:y2, x1:x2] += mask
            del tile_out
            empty_cache(device)
            bar.update(1)

    bar.close()
    return accum / weight.clamp(min=1e-8)


# ── Main inference dispatcher ─────────────────────────────────────────────────

def process_image(model: torch.nn.Module, tensor: torch.Tensor,
                  device: torch.device,
                  use_tile: bool, tile_size: int, overlap: int,
                  use_tta: bool) -> torch.Tensor:
    """
    Run inference with optional tiling and/or TTA.

    Tiling   → full-resolution output without memory spikes.
    TTA      → average over 4 flip augmentations; reduces directional
               artefacts and smooths noise.
    Combined → each augmentation is processed via tiling (best quality,
               most time).
    """

    def _infer(t: torch.Tensor) -> torch.Tensor:
        if use_tile:
            return run_tiled(model, t, tile_size, overlap, device)
        out = model(t.to(device)).cpu()
        empty_cache(device)
        return out

    if not use_tta:
        return _infer(tensor)

    results = []
    for aug, inv in TTA_AUGMENTATIONS:
        results.append(inv(_infer(aug(tensor))))
    return torch.stack(results).mean(0)


# ── Pre / post processing ─────────────────────────────────────────────────────

def output_path_for(src_name: str, output_dir: str) -> str:
    """Remap .heic/.heif → .jpg (Pillow cannot write HEIC)."""
    base, ext = os.path.splitext(src_name)
    if ext.lower() in ('.heic', '.heif'):
        src_name = base + '.jpg'
    return os.path.join(output_dir, src_name)


def load_and_preprocess(path: str, max_size: int | None) -> torch.Tensor:
    """
    Load → optional downscale → crop to ÷32 → normalise to [-1, 1].
    Closes the PIL image immediately after tensor conversion to free memory.
    """
    img = Image.open(path).convert('RGB')

    if max_size is not None:
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    tensor = transforms.ToTensor()(img)
    img.close()

    _, h, w = tensor.shape
    new_h, new_w = h - (h % 32), w - (w % 32)
    if new_h != h or new_w != w:
        tensor = transforms.CenterCrop((new_h, new_w))(tensor)

    tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(tensor)
    return tensor.unsqueeze(0)   # [1, 3, H, W]


def load_model(model_name: str, weights_path: str,
               device: torch.device) -> torch.nn.Module:
    model = MODEL_REGISTRY[model_name]()
    state = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state)
    del state
    model.to(device)
    model.eval()
    return model


# ── Entry point ───────────────────────────────────────────────────────────────

def infer():
    parser = argparse.ArgumentParser(description='Run inference with a trained enhancement model')
    parser.add_argument('--model',     type=str,  choices=list(MODEL_REGISTRY.keys()), required=True)
    parser.add_argument('--weights',   type=str,  required=True)
    parser.add_argument('--input',     type=str,  default='input')
    parser.add_argument('--output',    type=str,  default='output')
    parser.add_argument('--tile',      action='store_true',
                        help='Tiled inference for full-resolution processing')
    parser.add_argument('--tile_size', type=int,  default=1024,
                        help='Tile size in px, must be ÷32 (default: 1024)')
    parser.add_argument('--overlap',   type=int,  default=128,
                        help='Tile overlap in px (default: 128)')
    parser.add_argument('--tta',       action='store_true',
                        help='Test-Time Augmentation: average 4 flips')
    parser.add_argument('--max_size',  type=int,  default=None,
                        help='Downscale longer edge to this many px before inference')
    args = parser.parse_args()

    # Validate tile_size
    if args.tile_size % 32 != 0:
        parser.error(f'--tile_size must be divisible by 32 (got {args.tile_size})')

    os.makedirs(args.output, exist_ok=True)
    device = get_device()

    print(f"Device    : {device}")
    mode_parts = []
    if args.tile:
        mode_parts.append(f"tiled ({args.tile_size}px, {args.overlap}px overlap)")
    if args.tta:
        mode_parts.append("TTA ×4")
    if not mode_parts:
        mode_parts.append("standard")
    print(f"Mode      : {', '.join(mode_parts)}")
    if args.max_size:
        print(f"Max size  : {args.max_size}px")

    model = load_model(args.model, args.weights, device)
    print(f"Model     : {args.model}  ({args.weights})\n")

    image_files = sorted([
        f for f in os.listdir(args.input) if f.lower().endswith(SUPPORTED_EXTS)
    ])
    if not image_files:
        print(f"No images found in '{args.input}'.")
        return

    print(f"{len(image_files)} image(s) found in '{args.input}'\n")

    with torch.inference_mode():
        for fname in tqdm(image_files, desc=f"[{args.model}]"):
            try:
                tensor = load_and_preprocess(
                    os.path.join(args.input, fname), args.max_size
                )
            except Exception as e:
                print(f"\n  ⚠  Skipping {fname}: {e}")
                continue

            output  = process_image(model, tensor, device,
                                    args.tile, args.tile_size, args.overlap,
                                    args.tta)
            out_img = (output.clamp(-1, 1) + 1) / 2

            save_image(out_img.float().cpu(), output_path_for(fname, args.output))

            del tensor, output, out_img
            empty_cache(device)

    print(f"\n✓ Done. Results saved to '{args.output}'")


if __name__ == '__main__':
    infer()