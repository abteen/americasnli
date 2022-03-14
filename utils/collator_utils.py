from transformers import DataCollatorForLanguageModeling, DataCollatorForTokenClassification, DataCollatorWithPadding

def get_language_modeling_collator(config, tokenizer):
    return DataCollatorForLanguageModeling(tokenizer, **config['collator_settings']['init'])

def get_default_collator(config, tokenizer):
    return DataCollatorWithPadding(tokenizer, **config['collator_settings']['init'])

def get_collator(config, tokenizer):

    collator_type = config['collator_settings']['collator_type']

    handler = {
        'language_modeling' : get_language_modeling_collator,
        'default': get_default_collator,
    }

    return handler[collator_type](config, tokenizer)
