from transformers import ElectraForSequenceClassification
from utils_electra import ElectraClassificationHeadCustom


def get_last_dropout(model):
    if isinstance(model, ElectraForSequenceClassification):
        if isinstance(model.classifier, ElectraClassificationHeadCustom):
            return model.classifier.dropout2
        else:
            return model.classifier.dropout
    else:
        return model.dropout


def set_last_dropout(model, dropout):
    if isinstance(model, ElectraForSequenceClassification):
        if isinstance(model.classifier, ElectraClassificationHeadCustom):
            model.classifier.dropout2 = dropout
        else:
            model.classifier.dropout
    else:
        model.dropout = dropout
