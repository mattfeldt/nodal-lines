#!/bin/bash -x
#SBATCH --account=cstma
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=128
#SBATCH --output=eigenvalues_strong.out
#SBATCH --error=eigenvalues_strong.err
#SBATCH --time=00:25:00
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1

ml Python/3.12.3 SciPy-bundle matplotlib Stages/2025 GCC/13.3.0 OpenMPI/5.0.5  mpi4py/4.0.1
srun python Dahne_6_triangles_14_27_distance_1_1.py
#srun python Dahne_ellipses_6_higher_precision.py 
