#!/usr/bin/env python3
"""Download selected KITTI odometry image sequences using HTTP byte ranges.

Only the selected image_0 PNGs are extracted from the official grayscale ZIP;
the full 22 GB archive is never stored on disk. Calibration and ground truth
are fetched from the small official KITTI odometry archives.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request
import zipfile

GRAY_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip"
CALIB_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip"
POSES_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip"


class HttpRangeFile:
    """Small seekable read-only file that caches HTTP range blocks for zipfile."""

    def __init__(self, url: str, block_size: int = 8 * 1024 * 1024, cache_blocks: int = 4):
        self.url = url
        self.block_size = block_size
        self.cache_blocks = cache_blocks
        self.position = 0
        self.cache = OrderedDict()
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            self.size = int(response.headers["Content-Length"])
            accepts_ranges = response.headers.get("Accept-Ranges", "").lower()
            if accepts_ranges and accepts_ranges != "bytes":
                raise OSError(f"Server does not advertise byte ranges: {accepts_ranges}")

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            new_pos = offset
        elif whence == os.SEEK_CUR:
            new_pos = self.position + offset
        elif whence == os.SEEK_END:
            new_pos = self.size + offset
        else:
            raise ValueError("invalid whence")
        if new_pos < 0:
            raise ValueError("negative seek position")
        self.position = new_pos
        return self.position

    def _block(self, block_index):
        if block_index in self.cache:
            self.cache.move_to_end(block_index)
            return self.cache[block_index]
        start = block_index * self.block_size
        end = min(self.size, start + self.block_size) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        expected_bytes = end - start + 1
        last_error = None
        # Large public archives occasionally reset a single range request.
        # Retry transient failures rather than aborting a 20+ GB scan at the
        # final blocks; validate every block before adding it to the cache.
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206:
                        raise OSError(f"Range request returned HTTP {response.status}, expected 206")
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {start}-"):
                        raise OSError(f"Unexpected Content-Range: {content_range}")
                    data = response.read()
                if len(data) != expected_bytes:
                    raise OSError(
                        f"Short range response at byte {start}: "
                        f"received {len(data)} of {expected_bytes} bytes"
                    )
                break
            except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError) as exc:
                last_error = exc
                if attempt == 4:
                    raise OSError(
                        f"Range download failed for {self.url} at byte {start} "
                        f"after 5 attempts: {exc}"
                    ) from exc
                delay = min(2 ** attempt, 16)
                print(
                    f"Range request at byte {start} failed (attempt {attempt + 1}/5): "
                    f"{exc}; retrying in {delay}s",
                    flush=True,
                )
                time.sleep(delay)
        self.cache[block_index] = data
        while len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return data

    def read(self, size=-1):
        if self.position >= self.size:
            return b""
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        chunks = []
        remaining = size
        while remaining:
            block_index = self.position // self.block_size
            within = self.position % self.block_size
            block = self._block(block_index)
            take = min(remaining, len(block) - within)
            if take <= 0:
                break
            chunks.append(block[within:within + take])
            self.position += take
            remaining -= take
        return b"".join(chunks)


def extract_small_zip(url: str, wanted_prefixes: tuple[str, ...], destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
            shutil.copyfileobj(response, tmp)
            tmp.flush()
            with zipfile.ZipFile(tmp.name) as archive:
                for member in archive.infolist():
                    if any(member.filename.startswith(prefix) for prefix in wanted_prefixes):
                        archive.extract(member, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", default="dataset/dataset")
    parser.add_argument("--sequences", nargs="+", default=["00", "01", "02", "05"])
    args = parser.parse_args()
    sequences = sorted({f"{int(seq):02d}" for seq in args.sequences})
    if any(len(seq) != 2 or int(seq) > 10 for seq in sequences):
        parser.error("sequences must be KITTI training IDs 00 through 10")
    destination = Path(args.destination).resolve()
    image_root = destination / "sequences"
    image_root.mkdir(parents=True, exist_ok=True)

    print(f"Opening official KITTI grayscale archive with range requests: {GRAY_URL}", flush=True)
    with zipfile.ZipFile(HttpRangeFile(GRAY_URL)) as archive:
        wanted = tuple(f"dataset/sequences/{seq}/image_0/" for seq in sequences)
        members = [m for m in archive.infolist()
                   if any(m.filename.startswith(prefix) for prefix in wanted)
                   and m.filename.lower().endswith(".png")]
        if not members:
            raise RuntimeError("No selected image_0 PNG files found in official archive")
        expected = {seq: 0 for seq in sequences}
        for member in members:
            for seq in sequences:
                if member.filename.startswith(f"dataset/sequences/{seq}/image_0/"):
                    expected[seq] += 1
                    break
        print("Selected frame counts: " + ", ".join(f"{k}={v}" for k, v in expected.items()), flush=True)
        if any(count < 2 for count in expected.values()):
            raise RuntimeError(f"At least one selected sequence is incomplete: {expected}")
        for idx, member in enumerate(members, start=1):
            relative = Path(member.filename).relative_to("dataset")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            if idx % 500 == 0 or idx == len(members):
                print(f"Extracted {idx}/{len(members)} images", flush=True)

    print("Downloading sequence calibration files", flush=True)
    extract_small_zip(CALIB_URL, tuple(f"dataset/sequences/{seq}/calib.txt" for seq in sequences), destination.parent)
    print("Downloading ground-truth poses", flush=True)
    extract_small_zip(POSES_URL, tuple(f"dataset/poses/{seq}.txt" for seq in sequences), destination.parent)

    missing = []
    for seq in sequences:
        seq_dir = image_root / seq
        images = list((seq_dir / "image_0").glob("*.png"))
        if len(images) < 2 or not (seq_dir / "calib.txt").is_file():
            missing.append(seq)
        if not (destination / "poses" / f"{seq}.txt").is_file():
            missing.append(f"{seq}:poses")
    if missing:
        raise RuntimeError(f"Dataset extraction incomplete: {missing}")
    print(f"Dataset ready at {destination}; sequences={','.join(sequences)}", flush=True)


if __name__ == "__main__":
    main()
