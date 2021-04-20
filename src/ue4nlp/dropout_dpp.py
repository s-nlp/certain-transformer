import torch

from alpaca.uncertainty_estimator.masks import build_mask

from .dropout_mc import DropoutMC

import numpy as np
import time
import random

import logging

log = logging.getLogger(__name__)


class DropoutDPP(DropoutMC):
    dropout_id = -1

    def __init__(
        self,
        p: float,
        activate=False,
        mask_name="dpp",
        max_n=100,
        max_frac=0.4,
        coef=1.0,
    ):
        super().__init__(p=p, activate=activate)

        self.mask = build_mask(mask_name)
        self.reset_mask = False
        self.max_n = max_n
        self.max_frac = max_frac
        self.coef = coef

        self.curr_dropout_id = DropoutDPP.update()
        log.debug(f"Dropout id: {self.curr_dropout_id}")

    @classmethod
    def update(cls):
        cls.dropout_id += 1
        return cls.dropout_id

    def calc_mask(self, x: torch.Tensor):
        return self.mask(x, dropout_rate=self.p, layer_num=self.curr_dropout_id).float()

    def get_mask(self, x: torch.Tensor):
        return self.mask(x, dropout_rate=self.p, layer_num=self.curr_dropout_id).float()

    #         if not self.reset_mask:
    #             if len(x.shape) == 2:
    #                 return self.calc_mask(x)
    #             else:
    #                 mask = self.calc_mask(x.reshape(-1, x.shape[-1]))
    #                 while len(mask.shape) < len(x.shape):
    #                     mask = mask.unsqueeze(dim=0)

    #                 return mask
    #         else:
    #             self.mask.reset()

    #             self.calc_mask(x) # dry run
    #             for _ in range(random.randint(0, 100)):
    #                 self.calc_mask(x)

    #             return self.calc_mask(x)

    def calc_non_zero_neurons(self, sum_mask):
        frac_nonzero = (sum_mask != 0).sum(axis=-1).item() / sum_mask.shape[-1]
        return frac_nonzero

    def forward(self, x: torch.Tensor):
        if self.training:
            return torch.nn.functional.dropout(x, self.p, training=True)
        else:
            if not self.activate:
                return x

            sum_mask = self.get_mask(x)

            norm = 1.0
            i = 1
            frac_nonzero = self.calc_non_zero_neurons(sum_mask)
            # print('==========Non zero neurons:', frac_nonzero, 'iter:', i, 'id:', self.curr_dropout_id, '******************')
            # while i < 30:
            while i < self.max_n and frac_nonzero < self.max_frac:
                # while frac_nonzero < self.max_frac:
                mask = self.get_mask(x)

                # sum_mask = self.coef * sum_mask + mask
                sum_mask += mask
                i += 1
                # norm = self.coef * norm + 1

                frac_nonzero = self.calc_non_zero_neurons(sum_mask)
                # print('==========Non zero neurons:', frac_nonzero, 'iter:', i, '******************')

            # res = x * sum_mask / norm
            print("Number of masks:", i)
            res = x * sum_mask / i
            return res
