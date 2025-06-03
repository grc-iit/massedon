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
iobench

# Example:
# iobench 4M 20M sequential 50 posix test_file 2
# (IO depth has to be given; for posix it will be ignored)
```

---

## Notes

- When running OIBench, you must first create the file and set up the percent to 0 (for all APIs):

```bash
iobench 4M 20M sequential 0 cufile_async test_file 4
```

- **Be particular about `transfer_size` and `block_size`!**
  - Block size needs to be at least `io_depth * transfer_size`.

---

## NVIDIA GDS FS Installation Troubleshooting

If you face troubles, follow the [NVIDIA GDS Troubleshooting Guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#mofed-req-install).

1. **Turn off IOMMU:**
   - Set `IOMMU=OFF` in your kernel parameters.

2. **MOFED Installation:**
   - When installing, ensure the following flags are enabled: `gds`, `nvme`, `hvmeof`.

3. **DOCA-OFED:**
   - DOCA-OFED is an equivalent package of MLNX_OFED, providing the same functionality.
   - [DOCA Download URL](https://developer.nvidia.com/doca-downloads?deployment_platform=Host-Server&deployment_package=DOCA-Host&target_os=Linux&Architecture=x86_64&Profile=doca-all)
   - For Ubuntu:
     ```bash
     sudo apt install doca-ofed mlnx-fw-updater mlnx-nvme-dkms
     sudo update-initramfs -u -k $(uname -r)
     sudo reboot
     ```

4. **Check NVMe Device Support for GDS:**
   ```bash
   cat /sys/block/<nvme>/integrity/device_is_integrity_capable
   ```

5. **nvidia-fs.ko Driver:**
   - Starting with kernel driver 2.17.5, proprietary `nvidia.ko` is not supported with GDS.
   - Install the open RM driver.

6. **Verify GDS Installation:**
   ```bash
   /usr/local/cuda-<x>.<y>/gds/tools/gdscheck.py -p
   ```
