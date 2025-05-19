# massedon

# Dependenceis

## IoWarp

### On personal
```bash
spack install iowarp +encrypt +compress +mpiio +vfd +nocompile
```


### On delta
Use proper cuda
```bash
echo "module load cuda/12.4.0" >> ~/.bashrc
source ~/.bashrc
```

Launch iowarp install:
```bash
cd massedon
sbatch job_specs/iowarp_install.sbatch
```

## HermesShm

```bash
spack load iowarp
scspkg create hermes_shm
git clone https://github.com/iowarp/cte-hermes-shm.git
cd cte-hermes-shm
mkdir build
cd build
cmake ../ \
-DCMAKE_CUDA_ARCHITECTURES=80 \
-DHSHM_RPC_THALLIUM=ON \
-DHSHM_ENABLE_COMPRESS=ON \
-DHSHM_ENABLE_ENCRYPT=ON \
-DHSHM_ENABLE_CUDA=ON \
-DHSHM_ENABLE_ROCM=OFF \
-DHSHM_BUILD_TESTS=OFF \
-DHSHM_ENABLE_MPI=ON \
-DCMAKE_INSTALL_PREFIX=$(scspkg pkg root hermes_shm)
make -j8
make install
```

NOTE CMAKE_CUDA_ARCHITECTURES is important. On Delta it is 80.

# Environment

Must be loaded each time you start a terminal
```bash
spack load iowarp
module load hermes_shm
```

# Compiling 

```bash
scspkg create massedon
cd massedon
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$(scspkg pkg root massedon)
make -j32 install
```

# Usage
```bash
module load massedon
iobench #iobench 4M 20M sequential 50 posix test_file 2 - sample input
```
When running OIBench, you must first create the file and set up the percent to 0. (For all APIs)
Ex. iobench 4M 20M sequential 0 cufile_async test_file 4
