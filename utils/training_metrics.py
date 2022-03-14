import logging
from sklearn.metrics import f1_score, accuracy_score



def xnli_metrics(eval_pred):

    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    macro_f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    accuracy = accuracy_score(y_true=labels, y_pred=preds)

    # Uncomment to log metrics during training
    # logging.info('Accuracy during training: {}'.format(accuracy))
    # logging.info('Macro F1 during training: {}'.format(macro_f1))

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
    }


