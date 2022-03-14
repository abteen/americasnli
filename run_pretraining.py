import logging, os
from pprint import pformat
from datetime import datetime

from utils.experiment_utils import setup_experiment

from joblib import Memory

from utils.tokenizer_utils import get_tokenizer
from utils.model_utils import get_model
from utils.collator_utils import get_collator
from utils.dataset_utils import get_dataset
from utils.trainer_utils import get_trainer


from transformers import TrainingArguments



if __name__ == '__main__':

    args, config = setup_experiment()
    time = datetime.now().strftime("-%m-%d-%H:%M")
    logging.info('Start time: {}'.format(time))

    logging.info('Loading tokenizer')
    tokenizer = get_tokenizer(config)

    logging.info('Tokenizer: {}'.format(pformat(tokenizer)))

    logging.info('Loading pretraining data for following languages: {}'.format(pformat(config['target_language'])))

    target_language = config['target_language']
    logging.info('Current target language: {}'.format(target_language))

    if config['use_wandb']:
        import wandb
        wandb.init(project=config['experiment_name'])
    else:
        os.environ['WANDB_DISABLED'] = 'True'

    training_dataset = get_dataset(config=config, tokenizer=tokenizer)

    logging.info('Final loaded training data: {}'.format(training_dataset.__len__()))
    logging.info('Final training examples sample: {}'.format(pformat(training_dataset.examples[:5])))

    model = get_model(config)

    output_directory = os.path.join(config['output_directory'], config['experiment_name'])
    logging.info('Output directory: {}'.format(output_directory))
    training_arguments = TrainingArguments(output_dir=output_directory,
                                           **config['training_arguments'])

    collator = get_collator(config, tokenizer)

    trainer = get_trainer(
        config,
        model=model,
        args=training_arguments,
        train_dataset=training_dataset,
        data_collator=collator
    )

    if os.path.isdir(output_directory) and any('checkpoint-' in files for files in os.listdir(output_directory)):
        logging.info('Resuming training from checkpoint...')
        trainer.train(resume_from_checkpoint=config.check_resume_training)
    else:
        logging.info('Training from beginning...')
        trainer.train()

    model.save_pretrained(os.path.join(output_directory, 'final_model'))
    logging.info('Model saved in: {}'.format(output_directory))
    logging.info('-'*25 + 'Finished with language set: {}'.format(target_language) + '-'*25)




