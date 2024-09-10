## FLamE: Few-shot Learning from Natural Language Explanations

This repository provides an original implementation of [FLamE: Few-shot Learning from Natural Language Explanations](https://aclanthology.org/2023.acl-long.372/) by Yangqiaoyu Zhou, Yiming Zhang, Chenhao Tan.


### Overview

FLamE is a two-stage framework to effectively learn from explanations.

In the first stage, we prompt LLM to generate explanations conditioned on each label.

In the second stage, we train a prompt-based model to predict the label given both the original inputs and the generated explanations.

Please see more details in [our paper](https://aclanthology.org/2023.acl-long.372/). Here is a figure for the overview of our framework.

![Alt text](./flame.jpg?raw=true "FLamE Overview")


### Input format

Take e-SNLI dev set (k=16) for instance, we need three types of data.

The first is the original task data (`data/e-SNLI-k16/dev.jsonl`), which is natural language inference in our case. An example line in this file is:
```
{
    "id":"5960769829.jpg#3r1n",
    "label":"entailment",
    "premise":"A person in an orange kayak navigates white water rapids.",
    "hypothesis":"A PERSON IS KAYAKING IN ROUGH WATERS.",
    "human_explanation":"White water rapids implies ROUGH WATERS."
}
```

Second, we need the logits from the `no-explanation` baseline to train the ensemble model. We store the logits in `data/e-SNLI-k16/dev_logits.txt`. Here, each line has three logits numbers, corresponding the three labels. In particular, the labels are ["contradiction", "entailment", "neutral"], so the ordering of the three logits numbers need to follow the order in this list.
This label list is defined in `./pet/pvp.py` (`EsnliJSONLProcessor` class).

Last but not least, we need the LLM generated explanations.
An example in `data/e-SNLI-k16/phle_expl/dev_curie_phle.jsonl` is
```
{
    "0":"A person in an orange kayak cannot navigate white water rapids.",
    "1":"If the person is in an orange kayak, she must be in rough waters.",
    "2":"Not all rapids are rough waters."
}
```
In this dictionary, the keys correspond to the three labels in e-SNLI, and the values are the explanations generated based on the different label conditions. In particular, the labels are ["contradiction", "entailment", "neutral"] and the numbers 0, 1, and 2 are the corresponding indices of the labels in this list. 
This label list is defined in `./pet/pvp.py` (`EsnliJSONLProcessor` class).

### How to run the code?

Environment can be found in `environment.yml`.

The (example) shell scripts for running FLamE and other baselines are in `./shell_scripts`. They all call `cli.py` with different arguments for different set-ups. The example scripts are running on `e-SNLI-k16` dataset, which is stored in `./data`. Feel free to modify the script to suit your own datasets.

For instance, in order to run FLamE on e-SNLI dataset with k=16 and explain-then-predict method, run `sh ./shell_scripts/esnli_k16_flame_phTrue_davinci_explain-then-predict.sh`.
The result accuracy can be found in the `dev_ensemble*.txt` file.

For other baselines (`PET` and `RoBERTa`), the result accuracy can be found in the `dev_result*.txt` file.

Note that we did a hyperparameter sweep in the original paper:
* beta_list = [0.0, 0.25, 0.5, 0.75, 1.0]
* beta_lr_list = ['2e-2', '2e-3', '2e-4']

So you may need to try these different hyperparameters to reproduce the numbers in the paper.