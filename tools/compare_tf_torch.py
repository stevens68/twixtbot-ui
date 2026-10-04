#! /usr/bin/env python

"""Compare a converted PyTorch checkpoint against TensorFlow reference outputs."""

import argparse

import numpy as np

from src.backend.torchmodel import TwixtNet
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="reference/tf_reference.npz")
    parser.add_argument("--checkpoint", default="model/torch.pt")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    ref = np.load(args.reference)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
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

    pwin_ok = np.allclose(
        torch_pwin, ref["pwin"], rtol=args.rtol, atol=args.atol
    )
    policy_ok = np.allclose(
        torch_logits, ref["movelogits"], rtol=args.rtol, atol=args.atol
    )

    print(
        "pwin:       "
        f"{'PASS' if pwin_ok else 'FAIL'} "
        f"max_abs={np.max(np.abs(torch_pwin - ref['pwin'])):.6g}"
    )
    print(
        "movelogits: "
        f"{'PASS' if policy_ok else 'FAIL'} "
        f"max_abs={np.max(np.abs(torch_logits - ref['movelogits'])):.6g}"
    )

    if not (pwin_ok and policy_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
