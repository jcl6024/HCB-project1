#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --job-name=HCB_FiniteT_eq_beta=10_half_filling

cd $SLURM_SUBMIT_DIR

module load julia/1.11.2

SYSTEM=$SLURM_SUBMIT_DIR/System
time julia -O3 --threads 24 MDF_FiniteT_Eq.jl