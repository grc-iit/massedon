# massedon

# Dependenceis

```bash
spack install iowarp +nocompile
spack load iowarp
```

```bash
scspkg create hermes_shm
git clone https://github.com/iowarp/cte-hermes-shm.git
cd cte-hermes-shm
mkdir build
cd build
cmake ../ \
-DCMAKE_CUDA_ARCHITECTURES=native \
-DHSHM_RPC_THALLIUM=ON \
-DHSHM_ENABLE_COMPRESS=ON \
-DHSHM_ENABLE_ENCRYPT=ON \
-DHSHM_ENABLE_CUDA=OFF \
-DHSHM_ENABLE_ROCM=OFF \
-DBUILD_HSHM_TESTS=ON \
-DHSHM_ENABLE_MPI=ON
make -j8
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
cd massedon
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$(scspkg pkg root massedon)
make -j32 install
```

# Usage
```bash
module load massedon
iobench
```
