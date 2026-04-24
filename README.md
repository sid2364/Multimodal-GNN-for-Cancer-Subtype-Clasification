# A Multimodal Graph Neural Network Framework for Cancer Molecular Subtype Classification

This is the code for the paper, [A Multimodal Graph Neural Network Framework for Cancer Molecular Subtype Classification](https://arxiv.org/abs/2302.12838).

## Get started
The complete conda enviroment list is in `enviroment.yml` file.

## Demo
For classificaiton on BRCA dataset, use the following command.

```
python cancer_test.py --model gat --num_gene 100 --cancer_subtype True --omic_mode 4 --shuffle_index 0 --gene_gene True --mirna_gene True --mirna_mirna True --parallel True --l2 True --decoder False --poolsize 8 --edge_weight True --epochs 200 --train_ratio 0.7 --test_ratio 0.1
```

## Demo Data
Due the bandwidth limit of Git LFS, we made a mirror host of the demo data on [OneDrive](https://uconn-my.sharepoint.com/:f:/g/personal/bingjun_li_uconn_edu/Euizuh7l_m5DmCq9FX8Vum8BNKBG6k6dpF-RumBfNJrzOA?e=64PG9S).


# Other notes

## What is the data we are looking at?
we have the BRCA dataset only (BReast invasive CArcinoma)

total 981 patients, 4 molecular subtypes (classes) of breast cancer
BRCA subtypes Counts
LumA 529
LumB 197
Basal 175
Her2 80


"omic mode" specifies which data we are actually using (+ structure of graph, etc.)
mode 0: mRNA only
mode 1: miRNA only
mode 2: mRNA + miRNA
mode 3: mRNA + CNV
mode 4: mRNA + CNV + miRNA

mRNA data - gene expression (how active is each gene)
CNV data - copy number variation (how many copies of each gene, should apparently be 2 but can be more or less in cancer)
miRNA data - microRNA expression (microRNAs are small RNA molecules that can regulate gene expression, by binding to mRNA and stopping it from being translated into protein)

## Their pipeline
 
1. load mRNA, CNV, miRNA data (patient data)
2. pick top-N (1000?) highest variance genees (because this tells us which genes are the most interesting/differentiating)
3. filter out patient data to just those genes
4. build graph with genes as nodes, edges from intereactions (gene-gene, miRNA-gene, miRNA-miRNA)
5. do whole train/val/test split
6. get label data and filter to just patients who have the subtype labels
7. train models, eval, etc.
8. STONKS!

## Some commands + results

Run everything with BRCA dataset, GCN model, and 100 genes.
```
python cancer_test.py --model gcn --num_gene 100 --cancer_subtype True --omic_mode 4 --shuffle_index 0 --gene_gene True --mirna_gene True --mirna_mirna True --parallel True --l2 True --decoder False --poolsize 8 --edge_weight True --epochs 100 --train_ratio 0.7 --test_ratio 0.1
```

IDENTITY MATRIX ABLATION TEST (adj_matrix = identity):
              precision    recall  f1-score   support

           0       0.94      1.00      0.97        16
           1       0.70      0.64      0.67        11
           2       0.71      0.67      0.69        54
           3       0.19      0.22      0.21        18

    accuracy                           0.64        99
   macro avg       0.63      0.63      0.63        99
weighted avg       0.65      0.64      0.64        99

AUPRC: 0.7077

BIOLOGICAL:
              precision    recall  f1-score   support

           0       0.94      1.00      0.97        16
           1       0.70      0.64      0.67        11
           2       0.79      0.35      0.49        54
           3       0.25      0.67      0.36        18

    accuracy                           0.55        99
   macro avg       0.67      0.66      0.62        99
weighted avg       0.71      0.55      0.56        99

AUPRC: 0.6965



Run everything with BRCA dataset, GAT model, and 100 genes.
```python cancer_test.py --model gat --num_gene 100 --cancer_subtype True --omic_mode 4 --shuffle_index 0 --gene_gene True --mirna_gene True --mirna_mirna True --parallel True --l2 True --decoder False --poolsize 8 --edge_weight True --epochs 100 --train_ratio 0.7 --test_ratio 0.1
```

IDENTITY MATRIX ABLATION TEST (adj_matrix = identity):
     precision    recall  f1-score   support

           0       0.00      0.00      0.00        16
           1       0.00      0.00      0.00        11
           2       0.55      1.00      0.71        54
           3       0.00      0.00      0.00        18

    accuracy                           0.55        99
   macro avg       0.14      0.25      0.18        99
weighted avg       0.30      0.55      0.39        99

AUPRC: 0.2500


BIOLOGICAL:
              precision    recall  f1-score   support

           0       0.94      1.00      0.97        16
           1       0.70      0.64      0.67        11
           2       0.75      0.81      0.78        54
           3       0.15      0.11      0.13        18

    accuracy                           0.70        99
   macro avg       0.64      0.64      0.64        99
weighted avg       0.66      0.70      0.68        99

AUPRC: 0.7134

# A Multimodal Graph Neural Network Framework for Cancer Molecular Subtype Classification

This is the code for the paper, [A Multimodal Graph Neural Network Framework for Cancer Molecular Subtype Classification](https://arxiv.org/abs/2302.12838).

## Get started
The complete conda enviroment list is in `enviroment.yml` file.

## Demo
For classificaiton on BRCA dataset, use the following command.

```
python cancer_test.py --model gat --num_gene 100 --cancer_subtype True --omic_mode 4 --shuffle_index 0 --gene_gene True --mirna_gene True --mirna_mirna True --parallel True --l2 True --decoder False --poolsize 8 --edge_weight True --epochs 200 --train_ratio 0.7 --test_ratio 0.1
```

## Demo Data
Due the bandwidth limit of Git LFS, we made a mirror host of the demo data on [OneDrive](https://uconn-my.sharepoint.com/:f:/g/personal/bingjun_li_uconn_edu/Euizuh7l_m5DmCq9FX8Vum8BNKBG6k6dpF-RumBfNJrzOA?e=64PG9S).


# Other notes

## What is the data we are looking at?
we have the BRCA dataset only (BReast invasive CArcinoma)

total 981 patients, 4 molecular subtypes (classes) of breast cancer
BRCA subtypes Counts
LumA 529
LumB 197
Basal 175
Her2 80


"omic mode" specifies which data we are actually using (+ structure of graph, etc.)
mode 0: mRNA only
mode 1: miRNA only
mode 2: mRNA + miRNA
mode 3: mRNA + CNV
mode 4: mRNA + CNV + miRNA

mRNA data - gene expression (how active is each gene)
CNV data - copy number variation (how many copies of each gene, should apparently be 2 but can be more or less in cancer)
miRNA data - microRNA expression (microRNAs are small RNA molecules that can regulate gene expression, by binding to mRNA and stopping it from being translated into protein)

## Their pipeline
 
1. load mRNA, CNV, miRNA data (patient data)
2. pick top-N (1000?) highest variance genees (because this tells us which genes are the most interesting/differentiating)
3. filter out patient data to just those genes
4. build graph with genes as nodes, edges from intereactions (gene-gene, miRNA-gene, miRNA-miRNA)
5. do whole train/val/test split
6. get label data and filter to just patients who have the subtype labels
7. train models, eval, etc.
8. STONKS!

## Some commands + results

Run everything with BRCA dataset, GCN model, and 100 genes.
```
python cancer_test.py --model gcn --num_gene 100 --cancer_subtype True --omic_mode 4 --shuffle_index 0 --gene_gene True --mirna_gene True --mirna_mirna True --parallel True --l2 True --decoder False --poolsize 8 --edge_weight True --epochs 100 --train_ratio 0.7 --test_ratio 0.1
```

IDENTITY MATRIX ABLATION TEST (adj_matrix = identity):
              precision    recall  f1-score   support

           0       0.94      1.00      0.97        16
           1       0.70      0.64      0.67        11
           2       0.71      0.67      0.69        54
           3       0.19      0.22      0.21        18

    accuracy                           0.64        99
   macro avg       0.63      0.63      0.63        99
weighted avg       0.65      0.64      0.64        99

AUPRC: 0.7077

BIOLOGICAL:
              precision    recall  f1-score   support

           0       0.94      1.00      0.97        16
           1       0.70      0.64      0.67        11
           2       0.79      0.35      0.49        54
           3       0.25      0.67      0.36        18

    accuracy                           0.55        99
   macro avg       0.67      0.66      0.62        99
weighted avg       0.71      0.55      0.56        99

AUPRC: 0.6965



Run everything with BRCA dataset, GAT model, and 100 genes.
```python cancer_test.py --model gat --num_gene 100 --cancer_subtype True --omic_mode 4 --shuffle_index 0 --gene_gene True --mirna_gene True --mirna_mirna True --parallel True --l2 True --decoder False --poolsize 8 --edge_weight True --epochs 100 --train_ratio 0.7 --test_ratio 0.1
```

IDENTITY MATRIX ABLATION TEST (adj_matrix = identity):
     precision    recall  f1-score   support

           0       0.00      0.00      0.00        16
           1       0.00      0.00      0.00        11
           2       0.55      1.00      0.71        54
           3       0.00      0.00      0.00        18

    accuracy                           0.55        99
   macro avg       0.14      0.25      0.18        99
weighted avg       0.30      0.55      0.39        99

AUPRC: 0.2500


BIOLOGICAL:
              precision    recall  f1-score   support

           0       0.94      1.00      0.97        16
           1       0.70      0.64      0.67        11
           2       0.75      0.81      0.78        54
           3       0.15      0.11      0.13        18

    accuracy                           0.70        99
   macro avg       0.64      0.64      0.64        99
weighted avg       0.66      0.70      0.68        99

AUPRC: 0.7134

# Plans

## Todo
- Understand the steps
- Understand every step fo the code (pipline)
- Rewrite a simple baseline and see how it works (less nodes)
- Frame the question of our work (do we need GNN)
- Questions for the meeting
- (Exploring adjacent litterature)

## Baseline
- Understand data
- Illustrate data
- Issues and why
- Improvements
- Test leakage and generalization
    
## Extra
- Embeddings
- Improve ML pipline
- Improve CF
- Improve results 

