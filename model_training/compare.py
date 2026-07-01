import os
import argparse
from PIL import Image, ImageDraw, ImageFont

from pillow_heif import register_heif_opener
register_heif_opener()


SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.heic', '.heif')

# Refactored to a dictionary for easier dynamic lookup
PANELS = {
    'input':   {'label': 'Input (iPhone)', 'color': (45,  45,  45 )},
    'unet':    {'label': 'UNet',           'color': (25,  60,  100)},
    'pix2pix': {'label': 'Pix2Pix',        'color': (70,  30,  90 )},
}

LABEL_HEIGHT  = 52
DIVIDER_WIDTH = 3
DIVIDER_COLOR = (15, 15, 15)
TEXT_COLOR    = (240, 240, 240)
FONT_SIZE     = 26


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_font(size: int) -> ImageFont.ImageFont | None:
    """Try common macOS system fonts, fall back to PIL default."""
    candidates = [
        '/System/Library/Fonts/Helvetica.ttc',
        '/System/Library/Fonts/SFNSDisplay.ttf',
        '/System/Library/Fonts/SFNSText.ttf',
        '/Library/Fonts/Arial.ttf',
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return None


def make_label_bar(text: str, width: int, bg_color: tuple,
                   font: ImageFont.ImageFont | None) -> Image.Image:
    bar  = Image.new('RGB', (width, LABEL_HEIGHT), color=bg_color)
    draw = ImageDraw.Draw(bar)

    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    else:
        tw, th = len(text) * 8, 14

    draw.text(
        ((width - tw) // 2, (LABEL_HEIGHT - th) // 2),
        text, fill=TEXT_COLOR, font=font,
    )
    return bar


def find_output(input_name: str, output_dir: str) -> str | None:
    """
    Locate the processed output, accounting for HEIC → JPG rename.
    """
    base, ext = os.path.splitext(input_name)
    candidates = [input_name]
    if ext.lower() in ('.heic', '.heif'):
        candidates += [base + '.jpg', base + '.jpeg', base + '.JPG']
    for name in candidates:
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            return path
    return None


def resize_to_height(img: Image.Image, height: int) -> Image.Image:
    if img.height == height:
        return img
    scale = height / img.height
    return img.resize((int(img.width * scale), height), Image.LANCZOS)


# ── Core comparison builder ───────────────────────────────────────────────────

def build_comparison(paths: list[str], active_panels: list[dict],
                     target_height: int, font: ImageFont.ImageFont | None) -> Image.Image:
    images = [resize_to_height(Image.open(p).convert('RGB'), target_height)
              for p in paths]

    total_w = (sum(img.width for img in images)
               + DIVIDER_WIDTH * (len(images) - 1))
    total_h = LABEL_HEIGHT + target_height

    canvas = Image.new('RGB', (total_w, total_h), color=(12, 12, 12))

    x = 0
    for i, (img, panel) in enumerate(zip(images, active_panels)):
        if i > 0:
            canvas.paste(
                Image.new('RGB', (DIVIDER_WIDTH, total_h), DIVIDER_COLOR),
                (x, 0),
            )
            x += DIVIDER_WIDTH

        canvas.paste(make_label_bar(panel['label'], img.width, panel['color'], font), (x, 0))
        canvas.paste(img, (x, LABEL_HEIGHT))
        x += img.width

    return canvas


# ── Entry point ───────────────────────────────────────────────────────────────

def compare():
    parser = argparse.ArgumentParser(description='Side-by-side model output comparisons')
    parser.add_argument('--input',   type=str, default='input',
                        help='Folder with original input images')
    parser.add_argument('--unet',    type=str, default='output/unet',
                        help='Folder with UNet outputs')
    parser.add_argument('--pix2pix', type=str, default='output/pix2pix',
                        help='Folder with Pix2Pix outputs')
    parser.add_argument('--output',  type=str, default='output/comparisons',
                        help='Folder to save comparison images')
    parser.add_argument('--height',  type=int, default=1200,
                        help='Height of each panel in px (default: 1200)')
    parser.add_argument('--mode',    type=str, choices=['all', 'unet', 'pix2pix'], default='all',
                        help='Which outputs to include in the comparison (default: all)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    font = load_font(FONT_SIZE)

    input_files = sorted([
        f for f in os.listdir(args.input) if f.lower().endswith(SUPPORTED_EXTS)
    ])

    if not input_files:
        print(f"No images found in '{args.input}'.")
        return

    print(f"Found {len(input_files)} input image(s)\n")

    ok = skipped = 0
    for fname in input_files:
        input_path = os.path.join(args.input, fname)
        
        # Build dynamic lists of paths and panel configurations
        paths = [input_path]
        active_panels = [PANELS['input']]
        missing = []

        if args.mode in ('all', 'unet'):
            unet_path = find_output(fname, args.unet)
            if unet_path:
                paths.append(unet_path)
                active_panels.append(PANELS['unet'])
            else:
                missing.append('UNet')

        if args.mode in ('all', 'pix2pix'):
            pix2pix_path = find_output(fname, args.pix2pix)
            if pix2pix_path:
                paths.append(pix2pix_path)
                active_panels.append(PANELS['pix2pix'])
            else:
                missing.append('Pix2Pix')

        if missing:
            print(f"  ⚠  Skipping {fname}: {', '.join(missing)} output not found")
            skipped += 1
            continue

        base     = os.path.splitext(fname)[0]
        out_path = os.path.join(args.output, f'{base}_comparison_{args.mode}.jpg')

        try:
            img = build_comparison(paths, active_panels, args.height, font)
            img.save(out_path, quality=95)
            print(f"  ✓  {fname:40s} → {os.path.basename(out_path)}  ({img.width}×{img.height}px)")
            ok += 1
        except Exception as e:
            print(f"  ⚠  Failed {fname}: {e}")
            skipped += 1

    print(f"\n✓ {ok}/{len(input_files)} comparisons saved to '{args.output}'")


if __name__ == '__main__':
    compare()