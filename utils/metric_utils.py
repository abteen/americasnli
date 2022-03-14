from utils.training_metrics import xnli_metrics

def get_metric(metric):

    handler = {
        'nli' : xnli_metrics
    }

    return handler[metric]