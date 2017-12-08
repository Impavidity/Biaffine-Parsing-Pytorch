## Biaffine Parser

This repo is to replicate the model in the paper [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734). 

Currently, on the CoNLL09 English Dataset, I fall behind 1-2 points on UAS/LAS.


| CoNLL09-English | UAS | LAS |
|:-------:|:----:|:----:|
| Valid   | 93.52| 90.16|
| Test    | 94.22| 91.68|
| Original| 95.21| 93.20|


Need more tuning for these models.

### Quick Start

- This repo is based on my own framework [bpase](https://github.com/Impavidity/pbase). You can git clone that repo and then 
    ```text
    python setup.py install
    ```
    to install the library. There might be some dependency problems that I did not fix yet.
- Quick Start
    ```text
    python main.py 
    ``` 
    The model will be saved in *saves*.
- Test
    ```text
    python test.py --trained_model saves/name_of_model
    ```
    




