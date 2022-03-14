import logging

from transformers import XLMRobertaForMaskedLM, XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaForSequenceClassification
from transformers import BertModel, BertConfig, BertForTokenClassification


def get_pretrained_xlmr_mlm(config):

    model = XLMRobertaForMaskedLM.from_pretrained(**config['model_settings']['init'])

    return model

def get_pretrained_xlmr_seq_class(config):

    model = XLMRobertaForSequenceClassification.from_pretrained(**config['model_settings']['init'])

    return model


def get_model(config):

    model_type = config['model_settings']['model_type']

    handler = {
        'pretrained_xlmr_mlm': get_pretrained_xlmr_mlm,
        'pretrained_xlmr_seq_class': get_pretrained_xlmr_seq_class
    }

    return handler[model_type](config)
