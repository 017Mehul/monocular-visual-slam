#!/usr/bin/env python3
"""Convert docs/*.svg into PNG and GIF fallbacks.

Usage:
  python tools/convert_svgs.py [--out-dir docs] [--png] [--gif]

The script prefers `cairosvg` for SVG->PNG rendering and `Pillow` to create GIFs.
If `cairosvg` isn't installed, it will print instructions for installing it.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path


def ensure_deps():
    try:
        import cairosvg  # type: ignore
    except Exception:
        return False
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return False
    return True


def convert(svg_path: Path, out_dir: Path, make_png: bool = True, make_gif: bool = True) -> None:
    try:
        import cairosvg  # type: ignore
    except Exception:
        raise RuntimeError("cairosvg is required for SVG->PNG conversion")

    png_path = out_dir / (svg_path.stem + ".png")
    gif_path = out_dir / (svg_path.stem + ".gif")

    # Render PNG
    if make_png:
        print(f"Rendering {svg_path} -> {png_path}")
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))

    # Render GIF from PNG using Pillow
    if make_gif:
        try:
            from PIL import Image  # type: ignore

            if not png_path.exists():
                # fall back to rendering a PNG in-memory
                cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))

            img = Image.open(png_path).convert("RGBA")
            # convert to palette-based GIF to keep size small
            img = img.convert("P", palette=Image.ADAPTIVE)
            print(f"Saving GIF {gif_path}")
            img.save(gif_path, format="GIF")
        except Exception as e:
            print("Warning: GIF generation failed:", e)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="docs", help="Output directory for raster assets")
    ap.add_argument("--png", action="store_true", help="Only generate PNGs")
    ap.add_argument("--gif", action="store_true", help="Only generate GIFs (still requires PNG)")
    ap.add_argument("--force-install", action="store_true", help="Attempt to pip-install missing deps into current env")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    svgs = sorted(glob.glob("docs/*.svg"))
    if not svgs:
        print("No SVG files found in docs/ to convert.")
        return 0

    # Quick dependency check
    deps_ok = ensure_deps()
    if not deps_ok:
        print("Missing dependencies. To enable conversion, install:")
        print("  pip install cairosvg pillow")
        if args.force_install:
            print("Attempting pip install into current environment...")
            import subprocess

            subprocess.check_call([sys.executable, "-m", "pip", "install", "cairosvg", "pillow"])
            deps_ok = ensure_deps()
            if not deps_ok:
                print("Install failed. Aborting.")
                return 2
        else:
            return 1

    for s in svgs:
        svg_path = Path(s)
        try:
            convert(svg_path, out_dir, make_png=(not args.gif or not args.png), make_gif=(not args.png))
        except RuntimeError as e:
            print(e)
            return 3

    print("Done: raster assets written to", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
