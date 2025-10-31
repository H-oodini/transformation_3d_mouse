#!/usr/bin/env python3
"""
Capture a chessboard sequence on the Pi using libcamera-still.
"""

import os
import subprocess
from datetime import datetime

MOUNTPOINT = "/mnt/truenas"



def capture_sequence(out_root="captures", count=30, width=1640, height=1232, prefix="chessboard"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, f"{prefix}_leftCamera_{ts}_{width}x{height}")
    os.makedirs(out_dir, exist_ok=True)

    for i in range(count):
        filename = os.path.join(out_dir, f"{prefix}_{i+1:02d}.jpg")

        cmd = [
            "libcamera-still",
            "-o", filename,
            "--width", str(width),
            "--height", str(height),
            "--nopreview",
            "-t", "1"
        ]
        subprocess.run(cmd, check=True)

    # write a tiny manifest with metadata
    with open(os.path.join(out_dir, "manifest.txt"), "w") as f:
        f.write(f"width={width}\nheight={height}\ncount={count}\n")
    print(out_dir)
    
    
def sync_captures():
    src = os.path.expanduser("~/Desktop/captures")   # entire folder
    dst = "/mnt/truenas/FOR2591/jazmin/3D_reconstruction/"

    subprocess.run(["rsync", "-avh", src, dst], check=True)
    
    # remove data from pi after sync
    subprocess.run(["rsync", "-avh", "--remove-source-files", src, dst], check=True)
    
    return None

if __name__ == "__main__":
    capture_sequence()
    sync_captures()
