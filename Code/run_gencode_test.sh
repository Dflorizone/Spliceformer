#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=0-01:00:00
#SBATCH --output=slurm-%j.out

module load python/3.12
source ~/envs/spliceformer/bin/activate

cp /home/dflorizo/projects/def-hjabbari-ab/dflorizo/Spliceformer/Data/gencode_40k_dataset_test_.h5 $SLURM_TMPDIR/
python /home/dflorizo/projects/def-hjabbari-ab/dflorizo/Spliceformer/Code/gencode_test.py

cp $SLURM_TMPDIR/transformer_40k_test_gencode_predictions_250625.gz /home/dflorizo/projects/def-hjabbari-ab/dflorizo/Spliceformer/Data/