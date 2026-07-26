#!/usr/bin/env python3
"""Generate a StarDist 2D end-to-end instance parity fixture."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STARDIST_SRC = ROOT / "stardist"
MODEL_DIR = ROOT / "stardist" / "models" / "examples" / "2D_demo"
OUT = ROOT / "tests" / "fixtures" / "2d_demo_instances.npz"


def main():
    sys.path.insert(0, str(STARDIST_SRC))

    from stardist.models import StarDist2D

    model = StarDist2D(None, name="2D_demo", basedir=str(MODEL_DIR.parent))

    y = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    x = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    input_yx = 0.5 * y[:, None] + 0.5 * x[None, :]
    input_nhwc = input_yx[None, :, :, None].astype(np.float32)
    input_nchw = np.transpose(input_nhwc, (0, 3, 1, 2)).astype(np.float32)

    labels, polys = model.predict_instances(
        input_yx,
        axes="YX",
        normalizer=None,
        sparse=False,
        n_tiles=None,
        show_tile_progress=False,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        input_nchw=input_nchw,
        labels=labels.astype(np.uint32),
        coord=polys["coord"].astype(np.float32),
        points=polys["points"].astype(np.float32),
        prob=polys["prob"].astype(np.float32),
    )
    print(OUT)


if __name__ == "__main__":
    main()
