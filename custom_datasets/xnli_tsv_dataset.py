import torch, jsonlines, pandas, random, logging
from tqdm import trange

class xnliTSVDataset(torch.utils.data.Dataset):
    def __init__(self, file, tokenizer, max_len, lang, format, pred_loop_key=None, unseen=False):

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lang = lang
        self.unseen = unseen
        self.format = format

        if pred_loop_key is None:
            self.pred_loop_key = self.lang
        else:
            self.pred_loop_key = pred_loop_key

        self.examples = self.read_file(file)

        for i,ex in enumerate(self.examples[:3]):
            logging.info('Example {} for {} lang: {}'.format(i,lang,ex))

        logging.info('Pred loop key: {}'.format(self.pred_loop_key))
        logging.info('Number of examples: {}'.format(len(self.examples)))
        logging.info('Data loaded from: {}'.format(file))


        random.shuffle(self.examples)

        # logging.info('Pretokenizing...')
        # self.tokenized_examples = []
        # for i in trange(len(self.examples)):
        #     self.tokenized_examples.append(self.encode(i))

        logging.info('----------------------------------')


    def __getitem__(self, idx):
        # return self.tokenized_examples[idx]
        return self.encode(idx)

    def __len__(self):
        return len(self.examples)

    def read_file(self, file):

        inps = []
        labels_found = set()

        label2id = {
            'contradiction' : 0,
            'contradictory' : 0,
            'neutral' : 1,
            'entailment' : 2
        }


        with open(file, 'r') as f:
            for i, line in enumerate(f.readlines()):

                if i == 0:
                    print(line)
                    continue

                split = line.strip().split('\t')

                #Data input format: (premise, hypothesis, label)

                if self.format == 'mnli': #multinli_1.0_train.txt
                    inps.append((split[5], split[6], label2id[split[0]]))
                    labels_found.add(split[0])
                elif self.format == 'xnli': #xnli.dev.tsv, xnli.test.tsv
                    if split[0] == self.lang:
                        inps.append((split[6], split[7], label2id[split[1]]))
                        labels_found.add(split[1])
                elif self.format == 'mnli_translated': #mnli.train.es.tsv
                    inps.append((split[0], split[1], label2id[split[2]]))
                    labels_found.add(split[2])
                elif self.format == 'anli':
                    if split[1] == self.lang:
                        inps.append((split[2], split[3], label2id[split[4]]))
                        labels_found.add(split[4])
                elif self.format in ['translate-train']: #xnli_unseen.dev.tsv, translate_train.dev.tsv, train/${lang}.tsv
                    if split[0] == self.lang:
                        inps.append((split[1], split[2], label2id[split[3]]))
                        labels_found.add(split[3])



        return inps


    def encode(self, id):
        instance = self.examples[id]

        s1 = instance[0]
        s2 = instance[1]
        label = instance[2]

        enc = self.tokenizer(
            s1,s2,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_token_type_ids= True,
            return_tensors='pt'
        )

        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'token_type_ids': enc['token_type_ids'].squeeze(0),
            'labels': label
        }