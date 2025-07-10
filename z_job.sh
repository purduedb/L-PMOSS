#!/bin/bash
#SBATCH --account=csml-b         # Your project/account
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --cpus-per-task=4        # CPUs per task (threads)
#SBATCH --mem=64G                # Memory per node
#SBATCH --gres=gpu:1             # Number of GPUs
#SBATCH --time=01:00:00          # Max run time (hh:mm:ss)
#SBATCH --job-name=myjob         # Job name
#SBATCH --output=job_%j.out      # Standard output file
#SBATCH --error=job_%j.err       # Standard error file

# Load modules or activate environment if needed

module load conda
source activate pmoss
python run_dt_place.py --p "intel_skx_4s_8n" --mpath "None" --wl 12 --ecfg 100 --sidx 259 --rtg 2

# Run your program
python 
