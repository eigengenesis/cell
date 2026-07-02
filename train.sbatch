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

starts=$(date +"%s")
start=$(date +"%r, %m-%d-%Y")

module load conda
conda activate /depot/natallah/data/shourya/scDFM_env
export PYTHONPATH=./
export DEBUG_SYNC=${DEBUG_SYNC:-0}
if [ "$DEBUG_SYNC" = "1" ]; then export CUDA_LAUNCH_BLOCKING=1; fi

cd /depot/natallah/data/shourya/scDFM

COMMON="--batch_size=48 --devices=0 --model_type=origin --lr=5e-5 --steps=300000 \
--data_name=norman --d_model=128 --eta_min=1e-6 --fusion_method=differential_perceiver \
--split_method=additive --infer_top_gene=1000 --n_top_genes=5000 --result_path=./result/additive \
--perturbation_function=crisper --noise_type=Gaussian --mode=predict_y \
--deg_epsilon=0.1 --sinkhorn_eps=0.1 --sinkhorn_iters=50 \
--fold=1 --topk=30 --use_negative_edge --use_repo --use_wire"

python src/script/run.py $COMMON "$@"

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