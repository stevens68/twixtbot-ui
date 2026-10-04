import logging
from pathlib import Path

import numpy as np
import torch

from .. import constants as ct
from .nneval import NNEvaluater
from .torchmodel import TwixtNet


class TorchNNEvaluater:
    def __init__(self, model):
        self.logger = logging.getLogger(ct.LOGGER)
        self.model_path = model
        self.device = torch.device("cpu")
        checkpoint = torch.load(model, map_location=self.device, weights_only=True)
        config = dict(checkpoint["config"])
        config["channels_last"] = True
        self.model = TwixtNet(**config).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model = self.model.to(memory_format=torch.channels_last)
        self.model.eval()
        self.use_recents = self.model.use_recents

    def eval_one(self, nip):
        pegs, links, locs = nip.to_input_arrays(self.use_recents)
        inputs = [
            torch.from_numpy(x)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .contiguous(memory_format=torch.channels_last)
            .float()
            for x in (pegs, links, locs)
        ]
        with torch.inference_mode():
            pwin, movelogits = self.model(*inputs)
        return pwin.cpu().numpy(), movelogits.cpu().numpy()


class ParallelNNEvaluater:
    """Run TF and PyTorch together, return TF, and log numerical drift."""

    def __init__(self, tf_model, torch_model, atol=1e-5, rtol=1e-4,
                 strict=False):
        self.tf = NNEvaluater(tf_model)
        self.torch = TorchNNEvaluater(torch_model)
        if self.tf.use_recents != self.torch.use_recents:
            raise ValueError("TensorFlow/PyTorch recent-move inputs differ")
        self.use_recents = self.tf.use_recents
        self.atol = atol
        self.rtol = rtol
        self.strict = strict

    def eval_one(self, nip):
        tf_pwin, tf_logits = self.tf.eval_one(nip)
        torch_pwin, torch_logits = self.torch.eval_one(nip)

        pwin_error = float(np.max(np.abs(tf_pwin - torch_pwin)))
        logits_error = float(np.max(np.abs(tf_logits - torch_logits)))
        pwin_ok = np.allclose(tf_pwin, torch_pwin, rtol=self.rtol,
                              atol=self.atol)
        logits_ok = np.allclose(tf_logits, torch_logits, rtol=self.rtol,
                                atol=self.atol)

        if not (pwin_ok and logits_ok):
            self.tf.logger.warning(
                "TF/Torch drift: pwin_max_abs=%g logits_max_abs=%g",
                pwin_error, logits_error,
            )
            if self.strict:
                raise AssertionError(
                    f"TF/Torch parity failed: pwin={pwin_error:g}, "
                    f"logits={logits_error:g}"
                )
        else:
            self.tf.logger.debug(
                "TF/Torch parity: pwin_max_abs=%g logits_max_abs=%g",
                pwin_error, logits_error,
            )

        return tf_pwin, tf_logits


def _torch_model_path(model):
    """Resolve the PyTorch checkpoint for the configured TF model."""
    candidates = [
        Path(model).parent / "torch.pt",
        Path(__file__).resolve().parents[2] / "model" / "torch.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "PyTorch model not found. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def create_evaluator(backend, model):
    """Create the configured evaluator for a TensorFlow model path."""
    backend = backend.lower()
    if backend == "tensorflow":
        return None
    torch_model = _torch_model_path(model)
    if backend == "pytorch":
        return TorchNNEvaluater(torch_model)
    if backend == "parallel":
        return ParallelNNEvaluater(model, torch_model)
    raise ValueError(
        f"Unsupported NN backend: {backend!r}. "
        "Use tensorflow, pytorch, or parallel."
    )
