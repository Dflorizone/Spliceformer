#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000M
#SBATCH --time=0-06:00:00
#SBATCH --output=slurm-%j.out

module load python/3.12
source ~/envs/spliceformer/bin/activate

cp -r /home/dflorizo/projects/def-hjabbari-ab/dflorizo/Spliceformer/Data/ $SLURM_TMPDIR/
python /home/dflorizo/projects/def-hjabbari-ab/dflorizo/Spliceformer/Code/spliceAI_ensembl_test.py

cp $SLURM_TMPDIR/Data/spliceai_10k_test_ensembl_predictions_300625.csv.gz /home/dflorizo/projects/def-hjabbari-ab/dflorizo/Spliceformer/Data/