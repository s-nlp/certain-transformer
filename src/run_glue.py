import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import sys
import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
from tqdm import tqdm
import json
import numpy as np
from pathlib import Path
import random
import torch
import hydra

from utils_wandb import init_wandb, wandb
from utils_transformers_cached import (
    ElectraForSequenceClassificationCached,
    BertForSequenceClassificationCached,
)

from ue4nlp.dropout_mc import DropoutMC, activate_mc_dropout, convert_to_mc_dropout
from ue4nlp.dropout_dpp import DropoutDPP
from ue4nlp.text_classifier import TextClassifier

from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    GlueDataset
)
from transformers import ElectraForSequenceClassification

from utils_exps import set_last_dropout, get_last_dropout
from utils_electra import ElectraClassificationHeadCustom

from datasets import load_metric


import logging

log = logging.getLogger(__name__)


def convert_dropouts(model, ue_args):
    if ue_args.dropout_type == "MC":
        dropout_ctor = lambda p, activate: DropoutMC(
            p=ue_args.inference_prob, activate=False
        )
    elif ue_args.dropout_type == "DPP":

        def dropout_ctor(p, activate):
            return DropoutDPP(
                p=p,
                activate=activate,
                max_n=ue_args.dropout.max_n,
                max_frac=ue_args.dropout.max_frac,
                mask_name=ue_args.dropout.mask_name,
            )

    else:
        raise ValueError(f"Wrong dropout type: {ue_args.dropout_type}")

    if ue_args.dropout_subs == "last":
        set_last_dropout(model, dropout_ctor(p=ue_args.inference_prob, activate=False))

    elif ue_args.dropout_subs == "all":
        # convert_to_mc_dropout(model, {'Dropout': dropout_ctor})
        convert_to_mc_dropout(model.electra.encoder, {"Dropout": dropout_ctor})
    else:
        raise ValueError(f"Wrong ue args {ue_args.dropout_subs}")


def calculate_dropouts(model):
    res = 0
    for i, layer in enumerate(list(model.children())):
        module_name = list(model._modules.items())[i][0]
        layer_name = layer._get_name()
        if layer_name == "Dropout":
            res += 1
        else:
            res += calculate_dropouts(model=layer)

    return res


def freeze_all_dpp_dropouts(model, freeze):
    for layer in model.children():
        if isinstance(layer, DropoutDPP):
            if freeze:
                layer.mask.freeze(dry_run=True)
            else:
                layer.mask.unfreeze(dry_run=True)
        else:
            freeze_all_dpp_dropouts(model=layer, freeze=freeze)


def compute_metrics(is_regression, metric, p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    result = metric.compute(predictions=preds, references=p.label_ids)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()

    return result


def do_predict_eval(
    model, tokenizer, trainer, eval_dataset, train_dataset, metric, config, work_dir
):
    log.info("*** Evaluate ***")

    training_args = config.training

    true_labels = [example.label for example in eval_dataset]

    tagger = TextClassifier(
        model, tokenizer, training_args=training_args, trainer=trainer
    )

    preds, probs = tagger.predict(eval_dataset)

    ue_args = config.ue

    eval_results = {}
    eval_results["true_labels"] = true_labels
    eval_results["probabilities"] = probs.tolist()
    eval_results["answers"] = preds.tolist()
    eval_results["sampled_probabilities"] = []
    eval_results["sampled_answers"] = []

    log.info("******Perform stochastic inference...*******")

    log.info("Model before dropout replacement:")
    log.info(str(model))
    convert_dropouts(model, ue_args)
    log.info("Model after dropout replacement:")
    log.info(str(model))

    activate_mc_dropout(model, activate=True, random=ue_args.inference_prob)

    if ue_args.dropout_type == "DPP":
        log.info("**************Dry run********************")

        freeze_all_dpp_dropouts(model, freeze=True)

        dry_run_dataset = (
            eval_dataset if ue_args.dropout.dry_run_dataset == "eval" else train_dataset
        )
        tagger.predict(dry_run_dataset)

        freeze_all_dpp_dropouts(model, freeze=False)

        log.info("Done.")

    log.info("****************Start runs**************")
    eval_metric = metric

    set_seed(config.seed)
    random.seed(config.seed)
    for i in tqdm(range(ue_args.committee_size)):
        preds, probs = tagger.predict(eval_dataset)
        eval_results["sampled_probabilities"].append(probs.tolist())
        eval_results["sampled_answers"].append(preds.tolist())

        if ue_args.eval_passes:
            eval_score = eval_metric.compute(predictions=preds, references=true_labels)
            log.info(f"Eval score: {eval_score}")

    log.info("Done.")

    activate_mc_dropout(model, activate=False)

    with open(Path(work_dir) / "dev_inference.json", "w") as res:
        json.dump(eval_results, res)

    if wandb.run is not None:
        wandb.save(Path(work_dir) / "dev_inference.json")


def fix_task_name(task_name):
    return "sst2" if task_name == "sst-2" else task_name


def train_eval_glue_model(config, training_args, data_args, work_dir):
    ue_args = config.ue
    model_args = config.model

    # Set seed
    log.info(f"Seed: {config.seed}")
    set_seed(config.seed)
    random.seed(config.seed)

    mnli_mm = False
    if data_args.task_name == "mnli-mm":
        mnli_mm = True
        data_args.task_name = "mnli"

    try:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            num_labels = glue_tasks_num_labels[data_args.task_name]
        else:
            num_labels = 1

    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=config.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=config.cache_dir,
    )

    if ue_args.use_cache:
        if "electra" in model_args.model_name_or_path:  # TODO:
            model = ElectraForSequenceClassificationCached.from_pretrained(
                model_args.model_name_or_path,
                from_tf=False,
                config=model_config,
                cache_dir=config.cache_dir,
            )
            model.use_cache = True
            model.classifier = ElectraClassificationHeadCustom(model.classifier)
            log.info("Replaced ELECTRA's head")

        elif "bert" in model_args.model_name_or_path:
            model = BertForSequenceClassificationCached.from_pretrained(
                model_args.model_name_or_path,
                from_tf=False,
                config=model_config,
                cache_dir=config.cache_dir,
            )
            model.use_cache = True

        else:
            raise ValueError(
                f"{model_args.model_name_or_path} does not have a cached option."
            )

    else:
        if "electra" in model_args.model_name_or_path:
            model = ElectraForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=False,
                config=model_config,
                cache_dir=config.cache_dir,
            )

            model.classifier = ElectraClassificationHeadCustom(model.classifier)
            log.info("Replaced ELECTRA's head")
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=False,
                config=model_config,
                cache_dir=config.cache_dir,
            )

    print(model)

    train_dataset = None
    if config.do_train or (
        config.ue.dropout_type == "DPP" and config.ue.dropout.dry_run_dataset != "eval"
    ):
        train_dataset = GlueDataset(
            data_args, tokenizer=tokenizer, cache_dir=config.cache_dir
        )

    if config.do_train and config.data.subsample_perc > 0:
        indexes = list(range(len(train_dataset)))
        train_indexes = random.sample(
            indexes, int(len(train_dataset) * config.data.subsample_perc)
        )
        train_dataset = torch.utils.data.Subset(train_dataset, train_indexes)

    if mnli_mm:
        data_args = dataclasses.replace(data_args, task_name="mnli-mm")

    eval_dataset = (
        GlueDataset(
            data_args, tokenizer=tokenizer, mode="dev", cache_dir=config.cache_dir
        )
        if config.do_eval
        else None
    )

    metric_task_name = "sst2" if data_args.task_name == "sst-2" else data_args.task_name
    metric = load_metric(
        "glue", metric_task_name, keep_in_memory=True, cache_dir=config.cache_dir
    )  #
    metric_fn = lambda p: compute_metrics(is_regression, metric, p)

    training_args.save_steps = 0
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metric_fn,
    )

    if config.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path
            if os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.save_model(work_dir)
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(work_dir)

    if config.do_eval:
        do_predict_eval(
            model,
            tokenizer,
            trainer,
            eval_dataset,
            train_dataset,
            metric,
            config,
            work_dir,
        )


def update_config(cfg_old, cfg_new):
    for k, v in cfg_new.items():
        if k in cfg_old.__dict__:
            setattr(cfg_old, k, v)

    return cfg_old


@hydra.main(config_path=os.environ["HYDRA_CONFIG_PATH"])
def main(config):
    os.environ["WANDB_WATCH"] = "False"  # To disable Huggingface logging

    auto_generated_dir = os.getcwd()
    log.info(f"Work dir: {auto_generated_dir}")
    os.chdir(hydra.utils.get_original_cwd())

    wandb_run = init_wandb(auto_generated_dir, config)

    args_train = TrainingArguments(output_dir=auto_generated_dir)
    args_train = update_config(args_train, config.training)

    args_data = DataTrainingArguments(
        task_name=config.data.task_name, data_dir=config.data.data_dir
    )
    args_data = update_config(args_data, config.data)

    train_eval_glue_model(config, args_train, args_data, auto_generated_dir)


if __name__ == "__main__":
    main()
