import logging, os, sys

from datetime import datetime
from utils.experiment_utils import setup_experiment

from utils.metric_utils import get_metric
from utils.tokenizer_utils import get_tokenizer
from utils.model_utils import get_model
from utils.collator_utils import get_collator
from utils.dataset_utils import get_dataset

from overrides.eval_trainer import MultiEvalTrainer

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

if __name__ == '__main__':

    args, config = setup_experiment()

    tokenizer = get_tokenizer(config)

    time = datetime.now().strftime("-%m-%d-%H:%M")


    # Setup wandb
    if config['use_wandb']:
        import wandb

        wandb.init(project=config['experiment_name'])
    else:
        os.environ['WANDB_DISABLED'] = 'True'


    #Load model
    model = get_model(config)


    # Load Training Arguments
    output_directory = os.path.join(config['output_directory'], config['experiment_name'])
    final_model_directory = os.path.join(output_directory, 'final_model')
    if os.path.isdir(final_model_directory) and 'pytorch_model.bin' in os.listdir(final_model_directory):
        logging.info('Final model for this experiment exists, exiting without training')
        sys.exit(0)

    train_dataset = get_dataset(config.dataset_settings.train_dataset, tokenizer)
    eval_dataset = get_dataset(config.dataset_settings.eval_dataset, tokenizer)

    # test_dataset = get_dataset(config.test_dataset, tokenizer)

    training_arguments = TrainingArguments(output_dir=output_directory,
                                           **config['training_arguments'])

    collator = get_collator(config, tokenizer)

    metric = get_metric(config['task'])

    trainer = MultiEvalTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metric,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config['other_arguments'].early_stopping_patience)
        ]
    )

    if os.path.isdir(output_directory) and any('checkpoint-' in files for files in os.listdir(output_directory)):
        logging.info('Resuming training from checkpoint...')
        trainer.train(resume_from_checkpoint=config.check_resume_training)
    else:
        logging.info('Training from beginning...')
        trainer.train()

    model.save_pretrained(final_model_directory)
    logging.info('Model saved in: {}'.format(output_directory))


