# massedon

---

## Table of Contents
1. [Dependencies](#dependencies)
    - [IoWarp](#iowarp)
    - [HermesShm](#hermesshm)
2. [Environment Setup](#environment-setup)
3. [Compiling](#compiling)
4. [Usage](#usage)
5. [Notes](#notes)
6. [NVIDIA GDS FS Installation Troubleshooting](#nvidia-gds-fs-installation-troubleshooting)

---

## Dependencies

### IoWarp

#### On Personal Machine
```bash
spack install iowarp +encrypt +compress +mpiio +vfd +nocompile
```

#### On Delta
- Use proper CUDA:

```bash
echo "module load cuda/12.4.0" >> ~/.bashrc
source ~/.bashrc
```

- Launch IoWarp install:

```bash
cd massedon
sbatch job_specs/iowarp_install.sbatch
```

### HermesShm

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

> **Note:** `CMAKE_CUDA_ARCHITECTURES` is important. On Delta it is 80.

---

## Environment Setup

Must be loaded each time you start a terminal:

```bash
spack load iowarp
module load hermes_shm
```

---

## Compiling

```bash
scspkg create massedon
cd massedon
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$(scspkg pkg root massedon)
make -j32 install
```

---

## Usage

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

### 1. Turn off IOMMU
- Set `IOMMU=OFF` in your kernel parameters.
    1. Edit grub:
        ```bash
        sudo vi /etc/default/grub
        ```
    2. Add one of the following options to the `GRUB_CMDLINE_LINUX_DEFAULT` option:
        - For **AMD** CPU: `amd_iommu=off`
        - For **Intel** CPU: `intel_iommu=off`
        - Example: `GRUB_CMDLINE_LINUX_DEFAULT="console=tty0 amd_iommu=off"`
    3. Update grub and reboot:
        ```bash
        sudo update-grub
        sudo reboot
        ```
    4. After reboot, verify:
        ```bash
        cat /proc/cmdline
        ```
        It should show the options you added.

### 2. Install/Update MFT
- **Check the MFT version:**
    - For installed binaries:
        ```bash
        mft --version
        ```
    - For installed packages (.deb):
        ```bash
        dpkg -l | grep mft
        ```
- **If you don't have version > 4.32.0, run:**
    ```bash
    wget https://www.mellanox.com/downloads/MFT/mft-4.32.0-120-x86_64-deb.tgz
    tar -xvzf mft-4.32.0-120-x86_64-deb.tgz
    cd mft-4.32.0-120-x86_64-deb
    sudo ./install.sh
    ```

### 3. DOCA-OFED
- DOCA-OFED is an equivalent package of MLNX_OFED, providing the same functionality.
- [DOCA Download URL](https://developer.nvidia.com/doca-downloads?deployment_platform=Host-Server&deployment_package=DOCA-Host&target_os=Linux&Architecture=x86_64&Profile=doca-all)

    1. Download and install:
        ```bash
        wget https://www.mellanox.com/downloads/DOCA/DOCA_v3.0.0/host/doca-host_3.0.0-058000-25.04-ubuntu2404_amd64.deb
        sudo dpkg -i doca-host_3.0.0-058000-25.04-ubuntu2404_amd64.deb
        sudo apt-get update
        sudo apt-get -y install doca-all
        ```
    2. Install firmware updater:
        ```bash
        sudo apt install -y mlnx-fw-updater
        ```
    3. Load driver:
        ```bash
        sudo /etc/init.d/openibd restart
        ```
    4. Initialize MST:
        ```bash
        sudo mst restart
        ```
    5. For Ubuntu:
        ```bash
        sudo apt install doca-ofed mlnx-fw-updater mlnx-nvme-dkms
        sudo update-initramfs -u -k $(uname -r)
        sudo reboot
        ```

### 4. Install CUDA
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-9
```

### 5. Include all GDS packages
```bash
apt install cuda-toolkit
apt install nvidia-gds
reboot
```

### 6. Add to PATH
```bash
export PATH=${PATH}:/usr/local/cuda-12.9/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-12.9/lib64
```

### 7. Check NVMe Device Support for GDS
```bash
cat /sys/block/<nvme>/integrity/device_is_integrity_capable
```

### 8. Verify GDS Installation
```bash
/usr/local/cuda-12.9/gds/tools/gdscheck.py -p
```
