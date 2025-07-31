#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --mem=64gb
#SBATCH --time=2:00:00
#SBATCH --job-name=HCB_free_expansion_L=1000_N=51_V=1e-4

cd $SLURM_SUBMIT_DIR

module load julia/1.11.2

SYSTEM=$SLURM_SUBMIT_DIR/System
time julia -O3 --threads 24 MDF_ZeroT_NonEq_V2.jl