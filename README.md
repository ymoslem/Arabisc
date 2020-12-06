# Arabisc: Context-Sensitive Neural Spelling Checker

## Spelling Check Pre-trained Model

We are providing our pre-trained model for testing directly.

1. Download and unzip our spelling model at: https://arabic-spelling.s3-us-west-2.amazonaws.com/model-spell.zip
2. In the `data` folder of the current repository, unzip: **News-Multi.ar-en.ar.more.clean.zip**
3. Run the file **spelling-checker.py**:
```
python3 spelling-checker.py
```


## Training Spelling Check Model

This step is not required if you are going to use our pre-trained model. However, if you want to train a new spell check model, run the **train-dual-input.py** file as follows:
```
python3 train-dual-input.py <your_dataset_file_name.txt>
```

## Citation
If you are using or updating our spelling checking approach, Arabisc, please cite our paper, published in NLPTEA, ACCL 2020:
```
@inproceedings{moslem-etal-2020-arabisc,
    title = "Arabisc: Context-Sensitive Neural Spelling Checker",
    author = "Moslem, Yasmin  and
      Haque, Rejwanul  and
      Way, Andy",
    booktitle = "Proceedings of the 6th Workshop on Natural Language Processing Techniques for Educational Applications",
    month = dec,
    year = "2020",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.nlptea-1.2",
    pages = "11--19",
    abstract = "Traditional statistical approaches to spelling correction usually consist of two consecutive processes {---} error detection and correction {---} and they are generally computationally intensive. Current state-of-the-art neural spelling correction models usually attempt to correct spelling errors directly over an entire sentence, which, as a consequence, lacks control of the process, e.g. they are prone to overcorrection. In recent years, recurrent neural networks (RNNs), in particular long short-term memory (LSTM) hidden units, have proven increasingly popular and powerful models for many natural language processing (NLP) problems. Accordingly, we made use of a bidirectional LSTM language model (LM) for our context-sensitive spelling detection and correction model which is shown to have much control over the correction process. While the use of LMs for spelling checking and correction is not new to this line of NLP research, our proposed approach makes better use of the rich neighbouring context, not only from before the word to be corrected, but also after it, via a dual-input deep LSTM network. Although in theory our proposed approach can be applied to any language, we carried out our experiments on Arabic, which we believe adds additional value given the fact that there are limited linguistic resources readily available in Arabic in comparison to many languages. Our experimental results demonstrate that the proposed methods are effective in both improving the quality of correction suggestions and minimising overcorrection."
}
```
