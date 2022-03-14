import logging
from custom_datasets.pretraining_dataset import PretrainingDataset
from custom_datasets.xnli_tsv_dataset import xnliTSVDataset
from transformers import LineByLineTextDataset




def get_linebyline_dataset(config, tokenizer):

    train_dataset = LineByLineTextDataset(
        file_path=config['dataset_settings']['file_path'].format(config['target_language']),
        tokenizer=tokenizer,
        block_size=config['tokenizer_settings']['tokenization']['max_length']
    )

    return train_dataset


def get_mnli_dataset(config, tokenizer):

    logging.info('Loading dataset for {}'.format(config['language']))
    dataset = xnliTSVDataset(
        file=config['file'],
        tokenizer=tokenizer,
        max_len=config['max_sequence_length'],
        lang=config['language'],
        format=config['dataset_type']
    )

    return dataset


def get_translate_train_dataset(config, tokenizer):

    train_dataset = xnliTSVDataset(
        file=config['file'].format(config['language']),
        tokenizer=tokenizer,
        max_len=config['max_sequence_length'],
        lang=config['language'],
        format='translate-train'
    )

    return train_dataset


def get_eval_datasets(config, tokenizer):

    eval_datasets = []

    anli_lang = config.get('anli_language', None)
    if anli_lang:
        for lang in anli_lang:
            logging.info('Loading {} for evaluation'.format(lang))
            eval_datasets.append(
                xnliTSVDataset(
                    file=config['anli_dir'],
                    tokenizer=tokenizer,
                    max_len=config['max_sequence_length'],
                    lang=lang,
                    format='anli'
            ))

    xnli_lang = config.get('xnli_language', None)
    if xnli_lang:
        for lang in xnli_lang:
            logging.info('Loading {} for evaluation'.format(lang))
            eval_datasets.append(
                xnliTSVDataset(
                    file=config['xnli_dir'],
                    tokenizer=tokenizer,
                    max_len=config['max_sequence_length'],
                    lang=lang,
                    format='xnli'
                ))

    return eval_datasets

def get_eval_translate_train_dataset(config, tokenizer):

    lang = config['language']
    eval_datasets = []

    logging.info('Loading {} for evaluation'.format(lang))
    eval_datasets.append(
        xnliTSVDataset(
            file=config['anli_dir'],
            tokenizer=tokenizer,
            max_len=config['max_sequence_length'],
            lang=lang,
            format='anli'
        ))

    eval_datasets.append(
            xnliTSVDataset(
                file=config['translate_train_file'],
                tokenizer=tokenizer,
                max_len=config['max_sequence_length'],
                lang=lang,
                pred_loop_key='translate_train_' + lang,
                format='translate-train'
    ))

    return eval_datasets








def get_dataset(config, tokenizer):

    dataset_type = config['dataset_type']

    handler = {
        'line_by_line': get_linebyline_dataset,
        'eval': get_eval_datasets,
        'mnli': get_mnli_dataset,
        'mnli_translated': get_mnli_dataset,
        'translate_train': get_translate_train_dataset,
        'eval_translate_train': get_eval_translate_train_dataset
    }


    return handler[dataset_type](config, tokenizer)
