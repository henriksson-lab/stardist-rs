#!/usr/bin/env python3
"""Generate a StarDist 3D end-to-end instance parity fixture."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STARDIST_SRC = ROOT / "stardist"
MODEL_DIR = ROOT / "stardist" / "models" / "examples" / "3D_demo"
OUT = ROOT / "tests" / "fixtures" / "3d_demo_instances.npz"


def main():
    sys.path.insert(0, str(STARDIST_SRC))

    from stardist.models import StarDist3D

    model = StarDist3D(None, name="3D_demo", basedir=str(MODEL_DIR.parent))

    z = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    y = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    x = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    input_zyx = (
        0.25 * z[:, None, None] + 0.35 * y[None, :, None] + 0.40 * x[None, None, :]
    )
    input_ndhwc = input_zyx[None, :, :, :, None].astype(np.float32)
    input_ncdhw = np.transpose(input_ndhwc, (0, 4, 1, 2, 3)).astype(np.float32)

    labels, polys = model.predict_instances(
        input_zyx,
        axes="ZYX",
        normalizer=None,
        sparse=False,
        n_tiles=None,
        show_tile_progress=False,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        input_ncdhw=input_ncdhw,
        labels=labels.astype(np.uint32),
        dist=polys["dist"].astype(np.float32),
        points=polys["points"].astype(np.float32),
        prob=polys["prob"].astype(np.float32),
    )
    print(OUT)


if __name__ == "__main__":
    main()
