import torch, logging, random

from custom_datasets.pretraining_dataset_base import PretrainingDatasetBase

class PretrainingDataset(PretrainingDatasetBase):

    def __init__(self, training_data, tokenizer, **config):

        super().__init__(training_data, **config)

        self.tokenizer = tokenizer
        self.tokenization_settings = config['tokenizer_settings']

        print(self.tokenization_settings)

    def __getitem__(self, idx):

        instance = self.data[idx]

        enc = self.tokenizer(
            instance,
            **self.tokenization_settings['tokenization']
        )

        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0)
        }

