import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import Trainer

import numpy as np
import itertools
from tqdm import trange
import pickle
import json
import os

import logging

logger = logging.getLogger("sequence_tagger_auto")


class TextClassifier:
    def __init__(
        self,
        auto_model,
        bpe_tokenizer,
        max_len=192,
        pred_loader_args={"num_workers": 1},
        pred_batch_size=100,
        training_args=None,
        trainer=None,
    ):
        super().__init__()

        self._auto_model = auto_model
        self._bpe_tokenizer = bpe_tokenizer
        self._pred_loader_args = pred_loader_args
        self._pred_batch_size = pred_batch_size
        self._training_args = training_args
        self._trainer = trainer
        self._named_parameters = auto_model.named_parameters

    def predict(self, eval_dataset, evaluate=False, metrics=None):
        if metrics is None:
            metrics = []

        self._auto_model.eval()

        logits, _, metrics = self._trainer.predict(eval_dataset)
        probs = F.softmax(torch.tensor(logits), dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        print(metrics)

        return preds, probs
