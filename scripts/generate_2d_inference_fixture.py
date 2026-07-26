#!/usr/bin/env python3
"""Generate a StarDist 2D raw inference parity fixture.

The Rust parity test expects:
  tests/fixtures/2d_demo_inference.npz

This script intentionally uses the original Python implementation and bundled
demo weights as the source of truth.
"""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STARDIST_SRC = ROOT / "stardist"
MODEL_DIR = ROOT / "stardist" / "models" / "examples" / "2D_demo"
OUT = ROOT / "tests" / "fixtures" / "2d_demo_inference.npz"


def main():
    sys.path.insert(0, str(STARDIST_SRC))

    from stardist.models import StarDist2D

    model = StarDist2D(None, name="2D_demo", basedir=str(MODEL_DIR.parent))

    y = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    x = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    input_yx = 0.5 * y[:, None] + 0.5 * x[None, :]
    input_nhwc = input_yx[None, :, :, None].astype(np.float32)

    prob_nhwc, dist_nhwc = model.keras_model.predict(input_nhwc, verbose=0)
    input_nchw = np.transpose(input_nhwc, (0, 3, 1, 2)).astype(np.float32)
    prob_nchw = np.transpose(prob_nhwc, (0, 3, 1, 2)).astype(np.float32)
    dist_nchw = np.transpose(dist_nhwc, (0, 3, 1, 2)).astype(np.float32)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        input_nhwc=input_nhwc,
        input_nchw=input_nchw,
        prob_nhwc=prob_nhwc.astype(np.float32),
        prob_nchw=prob_nchw,
        dist_nhwc=dist_nhwc.astype(np.float32),
        dist_nchw=dist_nchw,
    )
    print(OUT)


if __name__ == "__main__":
    main()
