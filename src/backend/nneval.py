import logging
from pathlib import Path

import torch

from .. import constants as ct
from .torchmodel import TwixtNet


class NNEvaluater:
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
            torch.from_numpy(x).unsqueeze(0).float()
            for x in (pegs, links, locs)
        ]
        with torch.inference_mode():
            pwin, movelogits = self.model(*inputs)
        return pwin.cpu().numpy(), movelogits.cpu().numpy()
