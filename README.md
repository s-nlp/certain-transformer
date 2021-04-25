# Uncertainty Estimation for Transformer 

This repository is about Uncertainty Estimation (UE) for classification tasks on GLUE based on the Transformer models for NLP. Namely, the repository contain codes related to the the paper ["How certinaty is your Transformer?"](https://www.aclweb.org/anthology/2021.eacl-main.157/) at the EACL-2021 conference on NLP.

## What the paper is about? 

In this work, we consider the problem of uncertainty estimation for Transformer-based models. We investigate the applicability of uncertainty estimates based on dropout usage at the inference stage (Monte Carlo dropout). The series of experiments on natural language understanding tasks shows that the resulting uncertainty estimates improve the quality of detection of error-prone instances. Special attention is paid to the construction of computationally inexpensive estimates via Monte Carlo dropout and Determinantal Point Processes.


## More information and citation

You can learn more about the methods implemented in this repository in the following paper: 

Shelmanov, A., Tsymbalov, E., Puzyrev, D., Fedyanin, K., Panchenko, A., Panov, M. (2021): [How Certain is Your Transformer?](https://www.aclweb.org/anthology/2021.eacl-main.157/) In Proceeding of the 16th conference of the European Chapter of the Association for Computational Linguistics (EACL). 

If you found the materials presented in this paper and/or repository useful, please cite it as following:


```
@inproceedings{shelmanov-etal-2021-certain,
    title = "How Certain is Your {T}ransformer?",
    author = "Shelmanov, Artem  and
      Tsymbalov, Evgenii  and
      Puzyrev, Dmitri  and
      Fedyanin, Kirill  and
      Panchenko, Alexander  and
      Panov, Maxim",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.eacl-main.157",
    pages = "1833--1840",
    abstract = "In this work, we consider the problem of uncertainty estimation for Transformer-based models. We investigate the applicability of uncertainty estimates based on dropout usage at the inference stage (Monte Carlo dropout). The series of experiments on natural language understanding tasks shows that the resulting uncertainty estimates improve the quality of detection of error-prone instances. Special attention is paid to the construction of computationally inexpensive estimates via Monte Carlo dropout and Determinantal Point Processes.",
}
```


## Usage

```
HYDRA_CONFIG_PATH=../configs/sst2.yaml python ./run_glue.py
```
