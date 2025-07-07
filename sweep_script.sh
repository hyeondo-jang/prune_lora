#!/bin/sh
# SBATCH directives
#SBATCH -J gp-opt125m-normalize-grad  # Job name
#SBATCH -o ./out/%j.out  # Output file
##SBATCH -o ./out/%j.out
#SBATCH -t 3-00:00:00  # Run time (D-HH:MM:SS)

#### Select GPU
##SBATCH -p A100              # Partition
#SBATCH -p 3090              # Partition
##SBATCH -p A6000
#SBATCH --nodes=1            # Number of nodes
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4    # Number of CPUs
#SBATCH --gres=gpu:1         # Number of GPUs

cd $SLURM_SUBMIT_DIR

srun -I /bin/hostname
srun -I /bin/pwd
srun -I /bin/date

## Load modules
module purge
module load cuda/11.4.4
module load cudnn/cuda-12.1/8.9.7

## Python Virtual Environment
echo "Start"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export project_name="globalprune-admm" # W&B Project Name
export agent="7ogejp84" # W&B Sweep Agent ID
export env="gpa" # conda environment name

echo "source $HOME/anaconda3/etc/profile.d/conda.sh"
source /opt/anaconda3/2022.05/etc/profile.d/conda.sh    # Anaconda path


echo "conda activate $env"
conda activate $env    # Activate conda environment

# Run W&B Sweep Agent
srun wandb agent kwanheelee-postech/$project_name/$agent

date

echo "conda deactivate $env"
conda deactivate    # Deactivate environment

squeue --job $SLURM_JOBID

echo "##### END #####"
