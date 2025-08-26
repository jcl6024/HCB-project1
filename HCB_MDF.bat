#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --mem=64gb
#SBATCH --time=8:00:00
#SBATCH --job-name=HCB_finiteT_L=500_N=200_vary_beta
#SBATCH --output=%x.o%j

cd $SLURM_SUBMIT_DIR

module load julia/1.11.2

SYSTEM=$SLURM_SUBMIT_DIR/System
time julia -O3 --threads 24 MDF_FiniteT_Eq.jl