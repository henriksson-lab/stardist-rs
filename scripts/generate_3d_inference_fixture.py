#!/usr/bin/env python3
"""Generate a StarDist 3D raw inference parity fixture.

The Rust parity test expects:
  tests/fixtures/3d_demo_inference.npz

This script intentionally uses the original Python implementation and bundled
demo weights as the source of truth.
"""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STARDIST_SRC = ROOT / "stardist"
MODEL_DIR = ROOT / "stardist" / "models" / "examples" / "3D_demo"
OUT = ROOT / "tests" / "fixtures" / "3d_demo_inference.npz"


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

    prob_ndhwc, dist_ndhwc = model.keras_model.predict(input_ndhwc, verbose=0)
    input_ncdhw = np.transpose(input_ndhwc, (0, 4, 1, 2, 3)).astype(np.float32)
    prob_ncdhw = np.transpose(prob_ndhwc, (0, 4, 1, 2, 3)).astype(np.float32)
    dist_ncdhw = np.transpose(dist_ndhwc, (0, 4, 1, 2, 3)).astype(np.float32)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        input_ndhwc=input_ndhwc,
        input_ncdhw=input_ncdhw,
        prob_ndhwc=prob_ndhwc.astype(np.float32),
        prob_ncdhw=prob_ncdhw,
        dist_ndhwc=dist_ndhwc.astype(np.float32),
        dist_ncdhw=dist_ncdhw,
    )
    print(OUT)


if __name__ == "__main__":
    main()
