#!/bin/bash
set -e
mkdir -p logs
GRN_CSV=/depot/natallah/data/shourya/scDFM/data/norman/grn_edges.csv

# sbatch --job-name=e1_mmd_n \
#       train.sbatch --endpoint_loss=mmd          --gamma=0.1
# sbatch --job-name=e3_sinkhorn_n \
#       train.sbatch --endpoint_loss=sinkhorn     --gamma=0.1
# sbatch --job-name=e4_degsink_n \
#       train.sbatch --endpoint_loss=deg_sinkhorn --gamma=0.1
# sbatch --job-name=e6_degsink_grn \
#       train.sbatch --endpoint_loss=deg_sinkhorn --gamma=0.5 \
#       --grn_mask_path=$GRN_CSV

sbatch --job-name=e7_degsink_v \
      train.sbatch --endpoint_loss=deg_sinkhorn --gamma=0.1 --use_signed_edges