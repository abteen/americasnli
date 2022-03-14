import argparse, logging, os, torch

from transformers import XLMRobertaTokenizer
from transformers import XLMRobertaForSequenceClassification, XLMRobertaForMaskedLM
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback


from custom_datasets.xnli_tsv_dataset import xnliTSVDataset
from utils.training_metrics import xnli_metrics


def model_init():
    return XLMRobertaForSequenceClassification.from_pretrained(args.load_from_path,
                                                                 num_labels=3)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', default='data/anli_final/test/anli.test.tsv')
    parser.add_argument('--eval_file', default='data/anli_final/dev/anli.dev.tsv')
    parser.add_argument('--test_language', default='en')
    parser.add_argument('--test_format', default='anli')
    parser.add_argument('--eval_format', default='anli')
    parser.add_argument('--langs')

    parser.add_argument('--log_dir', default='logs/evaluation/')
    parser.add_argument('--xnli_dir', default='data/xnli/')
    parser.add_argument('--anli_dir', default='data/anli/')
    parser.add_argument('--output_dir')
    parser.add_argument('--experiment_name')
    parser.add_argument('--max_seq_len', default=256)
    parser.add_argument('--wandb_name')

    parser.add_argument('--load_from_path', default='xlm-roberta-base')

    args = parser.parse_args()

    if args.wandb_name:
        import wandb
        wandb.init(project=args.wandb_name, reinit=False)

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    logging.root.handlers = []
    # noinspection PyArgumentList
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("{}/{}.log".format(args.log_dir, args.experiment_name)),
            logging.StreamHandler()
        ]
    )

    logging.info('Number of GPUs available: {}'.format(torch.cuda.device_count()))


    xlmr_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

    model = XLMRobertaForSequenceClassification.from_pretrained(args.load_from_path,
                                                        num_labels=3)

    logging.info('Loading model from: {}'.format(args.load_from_path))

    trainer = Trainer(model=model,compute_metrics=xnli_metrics,args=TrainingArguments(output_dir='/rc_scratch/abeb4417/tempevaldir/', per_device_eval_batch_size=32))

    langs = [args.langs]
    test_format = args.test_format
    eval_format = args.eval_format
    test_file = args.test_file
    eval_file = args.eval_file

    if args.langs == 'anli':
        langs = ['aym','bzd','cni','gn','hch','nah','oto','quy','shp','tar']

    elif args.langs == 'xnli':
        langs = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
        test_format = 'xnli'
        eval_format = 'xnli'
        test_file = args.xnli_dir + 'xnli.test.tsv'
        eval_file = args.xnli_dir + 'xnli.dev.tsv'




    test_scores = []
    eval_scores = []


    for lang in langs:
        test_dataset = xnliTSVDataset(
            file=test_file,
            tokenizer=xlmr_tokenizer,
            max_len=args.max_seq_len,
            lang=lang,
            format=test_format
        )

        eval_dataset = xnliTSVDataset(
            file=eval_file,
            tokenizer=xlmr_tokenizer,
            max_len=args.max_seq_len,
            lang=lang,
            format=eval_format
        )

        eval_predictions = trainer.predict(eval_dataset)
        test_predictions = trainer.predict(test_dataset)

        logging.info('Language: {}'.format(lang))
        print(test_predictions)
        logging.info('Test accuracy: {:.2f}'.format(test_predictions.metrics['test_accuracy'] * 100))
        logging.info('Eval accuracy: {:.2f}'.format(eval_predictions.metrics['test_accuracy'] * 100))

        test_scores.append(test_predictions.metrics['test_accuracy'] * 100)
        eval_scores.append(eval_predictions.metrics['test_accuracy'] * 100)

        logging.info('---------------------------------')


    logging.info('All scores:')

    test_scores_str = ['{:.2f}'.format(score) for score in test_scores]
    eval_scores_str = ['{:.2f}'.format(score) for score in eval_scores]

    logging.info('Test: {}'.format(','.join(test_scores_str)))
    logging.info('Eval: {}'.format(','.join(eval_scores_str)))




    logging.info('Done')






