#!/bin/bash
#SBATCH --job-name=E_AI_triblock_370K
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=0:40:00
#SBATCH --mem-per-cpu=1G
#SBATCH --output=/scratch/jpdols/calvados/logs/%x_%j.out
#SBATCH --error=/scratch/jpdols/calvados/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate CALVADOS

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENMM_CPU_THREADS=$SLURM_CPUS_PER_TASK

cd /scratch/jpdols/calvados/runs/scale1/E_AI_triblock/6.0/370K/E_AI_triblock

python run.py --path .
