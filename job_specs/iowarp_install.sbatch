#!/bin/bash
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32   # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bekn-delta-cpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=install_iowarp
#SBATCH --time=01:15:00      # hh:mm:ss for the job
#SBATCH --constraint="scratch"
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
### GPU options ###
##SBATCH --gpus-per-node=0
##SBATCH --gpu-bind=none     # <- or closest
##SBATCH --mail-user=you@yourinstitution.edu
##SBATCH --mail-type="BEGIN,END" See sbatch or srun man pages for more email options

# cd ${HOME}
# git clone https://github.com/iowarp/iowarp-install.git

# cd ${HOME}
# git clone https://github.com/spack/spack.git
# cd spack
# git checkout tags/v0.22.2
# echo ". ${PWD}/share/spack/setup-env.sh" >> ~/.bashrc
# source ~/.bashrc

# cd ${HOME}
# spack repo add iowarp-install/iowarp-spack

spack install -y iowarp +mpiio +vfd +nocompile