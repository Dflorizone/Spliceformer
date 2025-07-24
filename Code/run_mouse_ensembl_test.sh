#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=20000M
#SBATCH --time=0-08:00:00
#SBATCH --output=slurm-%j.out

module load python/3.12
source ~/envs/spliceformer/bin/activate

cp -r /home/dflorizo/projects/def-hjabbari-ab/dflorizo/Spliceformer/Data/Mouse/v87/ $SLURM_TMPDIR/
python /home/dflorizo/projects/def-hjabbari-ab/dflorizo/Spliceformer/Code/mouse_ensembl_test.py

cp $SLURM_TMPDIR/v87/transformer_40k_test_mouse_ensembl_predictions_090725.csv.gz /home/dflorizo/projects/def-hjabbari-ab/dflorizo/Spliceformer/Data/Mouse/v87/