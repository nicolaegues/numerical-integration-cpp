
#!/bin/bash
#SBATCH --job-name=integration_cpp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:10:00
#SBATCH --account=chem033484
#SBATCH --partition=cpu

module load gcc/12.3.0
module load languages/python/3.12.3

export OMP_NUM_THREADS=16

./integration_methods
python integration_wrapper.py
