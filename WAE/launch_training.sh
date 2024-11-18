#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1              # 1 process
#SBATCH --cpus-per-task=4       # 4 CPUs
#SBATCH --mem=128GB              # 32 GB of memory
#SBATCH --time=24:00:00         # 1 hour run time
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1        # 1 node with 1 K40 GPU
#SBATCH --account=aoberai_286   # Account to charge resources to
module load gcc/11.3.0
module load conda/4.12.0
source activate tf2_env
python3 WAE_trainer.py \
        --dataname=training_dataset_slide0B44E01.npy \
	--slide_num=A8A805 \
        --n_epoch=3\
        --z_dim=50 \
        --lambda_param=100. \
        --n_train=30000000 \
        --batch_size=500 \
        --learn_rate=1e-5  \
        --lr_sched=True \
        --savefig_freq=2 \
        --save_suffix=_RCD_Denseblock_resources \
        --act_function=ReLU \
