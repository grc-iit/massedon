#!/bin/bash
#SBATCH --job-name="a.out_symmetric"
#SBATCH --output="a.out.%j.%N.out"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=64   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bekn-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --no-requeue
#SBATCH -t 01:30:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

# spack load ppi-chi-nettest python py-pip
# module load chi_nettest
# which jarvis
spack load iowarp
JARVIS_FIRST=$(jarvis sched hostfile build +slurm_host)
if [ "$JARVIS_FIRST" -eq 0 ]; then
    exit 0
fi
echo "On first node!!!"
jarvis ppl index load jarvis_massedon.delta.test_cufile_sync
