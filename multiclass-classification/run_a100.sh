#!/bin/sh

#SBATCH --account=aru
#SBATCH --partition=a100
#SBATCH --nodes=1 --ntasks=8
#SBATCH --time=3-03:00:00
#SBATCH --job-name="ftune1"
#SBATCH  --gres=gpu:a100-3g-20gb:1    
#SBATCH --mail-user=hnkmah001@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL

# --gres=gpu:a100-2g-10gb:1 #

# The cluster is configured primarily for OpenMPI and PMI. Use srun to launch parallel jobs if your code is parallel aware.
# To protect the cluster from code that uses shared memory and grabs all available cores the cluster has the following 
# environment variable set by default: OMP_NUM_THREADS=1
# If you feel compelled to use OMP then uncomment the following line:
# export OMP_NUM_THREADS=$SLURM_NTASKS

# NB, for more information read https://computing.llnl.gov/linux/slurm/sbatch.html



# Your science stuff goes here...
export CUDA_VISIBLE_DEVICES=$(ncvd)
module load software/TensorFlow-A100-GPU
# module load software/TensorFlow-2x-GPU
# python train_quat_cls_lsr_dense_vp_sampling.py --category=car --sigma=1000 --samples=50000
# python train_quat_cls_lsr_dense_vp_sampling.py --sigma=1000 --batch_size=64
python train_finetuning.py --batch_size=64
