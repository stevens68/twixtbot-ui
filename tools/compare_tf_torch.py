#! /usr/bin/env python

"""Compare a converted PyTorch checkpoint against TensorFlow reference outputs."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from src.backend.torchmodel import TwixtNet  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="reference/tf_reference.npz")
    parser.add_argument("--checkpoint", default="model/torch.pt")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    ref = np.load(args.reference)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    net = TwixtNet(**checkpoint["config"])
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()

    with torch.no_grad():
        pwin, movelogits = net(
            torch.from_numpy(ref["pegs"]).float(),
            torch.from_numpy(ref["links"]).float(),
            torch.from_numpy(ref["locs"]).float(),
        )

    torch_pwin = pwin.numpy()
    torch_logits = movelogits.numpy()
    # The reference is exported from per-position evaluation as (N, 1, ...),
    # while the batched PyTorch call returns (N, ...). Remove the singleton
    # position axis to avoid NumPy broadcasting batches against each other.
    ref_pwin = ref["pwin"][:, 0, :]
    ref_logits = ref["movelogits"][:, 0, :]

    pwin_ok = np.allclose(
        torch_pwin, ref_pwin, rtol=args.rtol, atol=args.atol
    )
    policy_ok = np.allclose(
        torch_logits, ref_logits, rtol=args.rtol, atol=args.atol
    )

    print(
        "pwin:       "
        f"{'PASS' if pwin_ok else 'FAIL'} "
        f"max_abs={np.max(np.abs(torch_pwin - ref_pwin)):.6g}"
    )
    print(
        "movelogits: "
        f"{'PASS' if policy_ok else 'FAIL'} "
        f"max_abs={np.max(np.abs(torch_logits - ref_logits)):.6g}"
    )

    if not (pwin_ok and policy_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
