#!/bin/bash -x
#SBATCH --account=cstma
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --output=eigenvalues_strong.out
#SBATCH --error=eigenvalues_strong.err
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1

ml Python/3.12.3 SciPy-bundle matplotlib Stages/2025 GCC/13.3.0 OpenMPI/5.0.5  mpi4py/4.0.1
#srun python Comparison_ellipses_5_hypotrochoid_fixed.py
srun python strong_scaling_eigenvalues_solution_0_holes.py
