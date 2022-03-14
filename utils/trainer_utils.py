from transformers import Trainer

def get_default_trainer(model, args, train_dataset, data_collator, **kwargs):
    return Trainer(
        model,
        args=args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )


def get_trainer(config, **kwargs):

    type = config['trainer_settings']['trainer_type']

    handler = {
        'default': get_default_trainer,
    }


    return handler[type](**kwargs)