#!/bin/bash	

#SBATCH --job-name=segformer
#SBATCH --output=output.out
#SBATCH --cpus-per-task=2
#SBATCH --time=140:00:00
#SBATCH --mem=8gb
#SBATCH --gres=gpu:Turing:4
#SBATCH --export=ALL

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

source ~/.bashrc
module load CUDA/11.8 CUDNN/8.6
conda activate py39

# python test_dataload.py --no-mps 
python segmentation_model.py --no-mps 

echo "Done: $(date +%F-%R:%S)"
