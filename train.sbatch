#!/bin/bash
#SBATCH -A pccr
#SBATCH -N 1
#SBATCH -p ai
#SBATCH -q preemptible
#SBATCH -t 24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=14
#SBATCH --mail-user=verma198@purdue.edu
#SBATCH --mail-type=FAIL
#SBATCH --job-name=scdfm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

starts=$(date +"%s")
start=$(date +"%r, %m-%d-%Y")

module load conda
conda activate /depot/natallah/data/shourya/scDFM_env
export DEBUG_SYNC=${DEBUG_SYNC:-0}
if [ "$DEBUG_SYNC" = "1" ]; then export CUDA_LAUNCH_BLOCKING=1; fi

VOID_CELL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCDFM_ROOT=${SCDFM_ROOT:-/depot/natallah/data/shourya/scDFM}
export PYTHONPATH="$VOID_CELL_DIR:$SCDFM_ROOT"

cd "$SCDFM_ROOT"

echo "VOID Cell source fingerprints"
git -C "$VOID_CELL_DIR" rev-parse HEAD
git -C "$VOID_CELL_DIR" status --short
sha256sum \
  "$VOID_CELL_DIR/config_flow.py" \
  "$VOID_CELL_DIR/instantiate_model.py" \
  "$VOID_CELL_DIR/layers.py" \
  "$VOID_CELL_DIR/blocks.py" \
  "$VOID_CELL_DIR/model.py" \
  "$VOID_CELL_DIR/sampling.py" \
  "$VOID_CELL_DIR/run.py"

COMMON="--batch_size=48 --devices=0 --model_type=origin --lr=5e-5 --steps=300000 \
--data_name=norman --d_model=128 --d_hid=2048 --encode_blocks=4 --think_steps=8 \
--eta_min=1e-6 --fusion_method=differential_perceiver --run_tag=void \
--split_method=additive --infer_top_gene=1000 --n_top_genes=5000 --result_path=./result/additive \
--perturbation_function=crisper --noise_type=Gaussian --mode=predict_y \
--endpoint_loss=deg_sinkhorn --gamma=0.1 --deg_epsilon=0.1 \
--sinkhorn_eps=0.1 --sinkhorn_iters=50 \
--fold=1 --topk=30 --use_negative_edge --use_signed_edges --use_repo --use_wire"

python -u "$VOID_CELL_DIR/run.py" $COMMON "$@"

ends=$(date +"%s")
end=$(date +"%r, %m-%d-%Y")
diff=$(($ends-$starts))
hours=$(($diff / 3600))
dif=$(($diff % 3600))
minutes=$(($dif / 60))
seconds=$(($dif % 60))
printf "\n\t===========Time Stamp===========\n"
printf "\tStart\t:$start\n\tEnd\t:$end\n\tTime\t:%02d:%02d:%02d\n" "$hours" "$minutes" "$seconds"
printf "\t================================\n\n"
sacct --jobs=$SLURM_JOBID --format=jobid,jobname,qos,nnodes,ncpu,maxrss,cputime,avecpu,elapsed
