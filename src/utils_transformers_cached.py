from typing import Optional

import torch
from torch.nn import CrossEntropyLoss

from transformers import (
    BertForSequenceClassification,
    ElectraForSequenceClassification,
)


class CachedInferenceMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_cache = False
        self.cache_size = None
        self.cache = dict()

    def empty_cache(self):
        self.cache.clear()

    def enable_cache(self):
        self.use_cache = True

    def disable_cache(self):
        self.use_cache = False
        self.empty_cache()

    def set_cache_size(self, size: Optional[int] = 25):
        self.cache_size = size

    @staticmethod
    def create_cache_key(tensor: torch.Tensor) -> int:
        return hash(frozenset(tensor.cpu().numpy().ravel()))

    def inference_body(
        self,
        body,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        inputs_embeds,
        output_attentions,
        output_hidden_states,
        return_dict,
    ):
        cache_key = self.create_cache_key(input_ids)

        if not self.use_cache or cache_key not in self.cache:
            hidden_states = body(
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                head_mask,
                inputs_embeds,
                output_attentions,
                output_hidden_states,
                return_dict,
            )

            if self.use_cache and (
                self.cache_size is None or len(self.cache) < self.cache_size
            ):
                self.cache[cache_key] = tuple(o.detach().cpu() for o in hidden_states)
        else:
            hidden_states = tuple(o.cuda() for o in self.cache[cache_key])

        return hidden_states


class ElectraForSequenceClassificationCached(
    CachedInferenceMixin, ElectraForSequenceClassification
):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        discriminator_hidden_states = self.inference_body(
            self.electra,
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = discriminator_hidden_states[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


class BertForSequenceClassificationCached(
    CachedInferenceMixin, BertForSequenceClassification
):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.inference_body(
            self.bert,
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
