#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --job-name=HCB_free_expansion_L=1000_N=101_V=2.6x1e-5

cd $SLURM_SUBMIT_DIR

module load julia/1.11.2

SYSTEM=$SLURM_SUBMIT_DIR/System
time julia -O3 --threads 24 MDF_ZeroT_NonEq.jl