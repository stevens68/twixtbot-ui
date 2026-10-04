import logging

import torch

from .. import constants as ct


class TorchNNEvaluater:

    def __init__(self, model):
        self.logger = logging.getLogger(ct.LOGGER)

        self.model_path = model
        self.device = torch.device("cpu")

        self.model = None

    def eval_one(self, nip):
        raise NotImplementedError
