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
    title = "{A}rabisc: Context-Sensitive Neural Spelling Checker",
    author = "Moslem, Yasmin  and
      Haque, Rejwanul  and
      Way, Andy",
    booktitle = "Proceedings of the Sixth Workshop of Natural Language Processing Techniques for Educational Applications (NLPTEA)",
    month = dec,
    year = "2020",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics (ACL)",
    pages = "11--19"
}
```
