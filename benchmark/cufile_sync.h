#ifndef CUFILE_SYNC_H
#define CUFILE_SYNC_H

#include <cuda_runtime.h>

#include "iobench.h"

class CufileSyncIoBench : public IoBench {
 public:
};

void cufile_io(const std::string& filename, size_t transfer_size,
               size_t block_size, IOType io_type, IOPattern io_pattern) {
  CUfileError_t status = cuFileDriverOpen();
  if (status.err != CU_FILE_SUCCESS) {
    return;
  }
  CUfileHandle_t fh;
  CUfileDescr_t params;
  memset(&params, 0, sizeof(params));
  int fd = open64(filename.c_str(), O_RDWR | O_CREAT, 0666);
  if (fd < 0) {
    int err = errno;
    std::cerr << "Error opening file: " << filename << " - " << strerror(err)
              << std::endl;
    return;
  }
  params.handle.fd = fd;
  params.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  status = cuFileHandleRegister(&fh, &params);
  if (status.err != CU_FILE_SUCCESS) {
    std::cerr << "cuFileHandleRegister error: " << status.err << std::endl;
    return;
  }

  char* device_buffer;
  cudaMalloc(&device_buffer, transfer_size);
  cudaStreamSynchronize(0);
  status = cuFileBufRegister(device_buffer, transfer_size, 0);
  if (status.err != CU_FILE_SUCCESS) {
    std::cerr << "cuFileBufRegister error: " << status.err << std::endl;
  }

  size_t num_transfers = block_size / transfer_size;
  for (size_t i = 0; i < num_transfers; ++i) {
    size_t offset =
        (io_pattern == IOPattern::RANDOM) ? distrib(gen) : i * transfer_size;
    ssize_t ret = cuFileWrite(fh, device_buffer, transfer_size, offset, 0);
    if (ret < 0) {
      std::cerr << "cuFileWrite error: " << ret << std::endl;
      break;
    }
  }
  cudaFree(device_buffer);
  cuFileHandleDeregister(fh);
}

#endif