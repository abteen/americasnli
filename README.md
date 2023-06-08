## AmericasNLI: Evaluating Zero-shot Natural Language Understanding of Pretrained Multilingual Models in Truly Low-resource Languages

This repository contains the code for AmericasNLI. Configurations which correspond to various aspects of a training run can be found in the configs folder. The scripts
folder contains bash scripts which combine the correct configurations to reproduce experiments from the paper. 

## Additional Data
Due to large file sizes, the data used for translation based approaches, and augmented pretraining can be found here: https://o365coloradoedu-my.sharepoint.com/:u:/g/personal/abeb4417_colorado_edu/EZJ2s8yKqUNGuxceT_nu6MMBDnfP9f-UCYrcGU5WiS_CIg

## Citation

If you use this work, please use the following citation:
```
@inproceedings{ebrahimi-etal-2022-americasnli,
    title = "{A}mericas{NLI}: Evaluating Zero-shot Natural Language Understanding of Pretrained Multilingual Models in Truly Low-resource Languages",
    author = "Ebrahimi, Abteen  and
      Mager, Manuel  and
      Oncevay, Arturo  and
      Chaudhary, Vishrav  and
      Chiruzzo, Luis  and
      Fan, Angela  and
      Ortega, John  and
      Ramos, Ricardo  and
      Rios, Annette  and
      Meza Ruiz, Ivan Vladimir  and
      Gim{\'e}nez-Lugo, Gustavo  and
      Mager, Elisabeth  and
      Neubig, Graham  and
      Palmer, Alexis  and
      Coto-Solano, Rolando  and
      Vu, Thang  and
      Kann, Katharina",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.435",
    pages = "6279--6299",
    abstract = "Pretrained multilingual models are able to perform cross-lingual transfer in a zero-shot setting, even for languages unseen during pretraining. However, prior work evaluating performance on unseen languages has largely been limited to low-level, syntactic tasks, and it remains unclear if zero-shot learning of high-level, semantic tasks is possible for unseen languages. To explore this question, we present AmericasNLI, an extension of XNLI (Conneau et al., 2018) to 10 Indigenous languages of the Americas. We conduct experiments with XLM-R, testing multiple zero-shot and translation-based approaches. Additionally, we explore model adaptation via continued pretraining and provide an analysis of the dataset by considering hypothesis-only models. We find that XLM-R{'}s zero-shot performance is poor for all 10 languages, with an average performance of 38.48{\%}. Continued pretraining offers improvements, with an average accuracy of 43.85{\%}. Surprisingly, training on poorly translated data by far outperforms all other methods with an accuracy of 49.12{\%}.",
}

```
