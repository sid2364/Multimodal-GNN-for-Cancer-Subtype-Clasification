#!/bin/bash

gene_nums=(200 500 700)

for num_gene in "${gene_nums[@]}"; do

    echo "===== running gcn with num_gene=$num_gene"
    python cancer_test.py --model gcn --num_gene $num_gene --cancer_subtype True --omic_mode 4 --shuffle_index 0 --gene_gene True --mirna_gene True --mirna_mirna True --parallel True --l2 True --decoder False --poolsize 8 --edge_weight True --epochs 100 --train_ratio 0.7 --test_ratio 0.1

    echo "===== running gat with num_gene=$num_gene"
    python cancer_test.py --model gat --num_gene $num_gene --cancer_subtype True --omic_mode 4 --shuffle_index 0 --gene_gene True --mirna_gene True --mirna_mirna True --parallel True --l2 True --decoder False --poolsize 8 --edge_weight True --epochs 100 --train_ratio 0.7 --test_ratio 0.1

    echo "===== running baseline with num_gene=$num_gene"
    python cancer_test.py --model baseline --cancer_subtype True --specific_type brca --omic_mode 4 --num_gene $num_gene --epochs 100

done

echo "done!"
