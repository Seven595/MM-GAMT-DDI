# MM-GAMT-DDI
Codes for "MM-GANN-DDI: Multimodal Graph-Agnostic Neural Networks for Predicting Drug-Drug Interaction Events".

**MM-GANN-DDI** is a tool than can predict DDI types based on multimodal GNNs with meta training mechanism.

## Environment Requirements:

```shell
1. Python == 3.7.5
2. PyTorch == 1.12.0
3. Rdkit
4. PyTorch_Geometric == 2.1.0
```

## To test on Dataset DB-v1 (from DrugBank):

```shell
cd multi-meta-DB-v1
# for task 1:
python dataPrepare-389-task1.py
# for task 2:
python dataPrepare-389-cov.py
# for task 3:
python datPrepare-389-task3-improve.py
```



## To test on Dataset DB-v2 (from DrugBank):

```shell
cd multi-meta-DB-v2
# for task 1:
python dataPrepare-570-task1.py
# for task 2:
python dataPrepare-570-cov-v4.py
# for task 3:
python dataPrepare-570-task3-v3.py
```

