#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1              # 1 process
#SBATCH --cpus-per-task=4       # 4 CPUs
#SBATCH --mem=32GB              # 32 GB of memory
#SBATCH --time=24:00:00         # 1 hour run time
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1        # 1 node with 1 K40 GPU
#SBATCH --account=aoberai_286   # Account to charge resources to

module load gcc/11.3.0
module load conda/4.12.0  
source activate tf2_env
python3 main.py \
        --problem=example1 \
        --max_epochs=600001 \
        --repetition=1 \
        --loss_type=Oberai
