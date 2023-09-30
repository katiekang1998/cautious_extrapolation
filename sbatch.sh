#!/bin/bash
#SBATCH --job-name=cautious_extrapolation
#SBATCH --open-mode=append
#SBATCH --output=logs/out/%x_%j.txt
#SBATCH --error=logs/err/%x_%j.txt
#SBATCH --time=120:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:TITAN:1
#SBATCH --account=co_rail
#SBATCH --partition=savio3_gpu
#SBATCH --qos=rail_gpu3_normal

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))

module load gnu-parallel

export PROJECT_DIR="/global/scratch/users/$USER/cautious_extrapolation/cautious_extrapolation"

run_singularity ()
{
singularity exec --nv --userns --writable-tmpfs -B /usr/lib64 -B /var/lib/dcv-gl --overlay /global/scratch/users/katiekang1998/overlay-50G-10M.ext3:ro /global/scratch/users/katiekang1998/singularity/cudagl11.5-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "
    source /ext3/env.sh
    source ~/.bashrc
    export WANDB_API_KEY=
    conda activate cifar10
    export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1
    XLA_PYTHON_CLIENT_PREALLOCATE=false python $PROJECT_DIR/CIFAR10/train.py \
        --data-loc=brc \
        --seed=$1 \
        --train_type=xent \
"
}


export -f run_singularity
parallel --delay 20 --linebuffer -j 2 run_singularity {1} ::: 0 1
