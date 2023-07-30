#!/bin/bash
#SBATCH --account=rokabe		# username to associate with job
#SBATCH --job-name=diff		# a desired name to appear alongside job ID in squeue
#SBATCH --gres=gpu:1 			# number of GPUs (per node)
#SBATCH --time=3-23:00			# time (DD-HH:MM)
#SBATCH --output="slurm/%x_%j.out"		# output file where all text printed to terminal will be stored
					# current format is set to "job-name_jobID.out"
nice python scripts/sample.py --model_path /home/rokabe/data2/generative/hydra/singlerun/2023-07-27/diff_mp20_1  --save_path /home/rokabe/data2/generative/hydra/singlerun/2023-07-27/diff_mp20_1 --formula AlFeClBrF --num_evals 20