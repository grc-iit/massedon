# massedon

# Dependenceis

## CUDA 
```
cd ~/Downloads
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt -y install cuda-toolkit
sudo apt -y install cuda-drivers
sudo apt-get install nvidia-gds
```

Create cuda module
```
scspkg create cuda
scspkg env prepend cuda PATH /usr/local/cuda/gds/tools /usr/local/cuda/bin
scspkg env prepend cuda CPATH /usr/local/cuda/targets/x86_64-linux/include
scspkg env prepend cuda LIBRARY_PATH /usr/local/cuda/targets/x86_64-linux/lib
scspkg env prepend cuda LD_LIBRARY_PATH /usr/local/cuda/targets/x86_64-linux/lib
```

## IoWarp

### On personal
```bash
spack install iowarp +nocompile
```


### On delta
Use proper cuda
```bash
echo "module load cuda/12.4.0" >> ~/.bashrc
source ~/.bashrc
```

Launch iowarp install:
```bash
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
