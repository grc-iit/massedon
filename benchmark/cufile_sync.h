// #ifndef CUFILE_SYNC_H
// #define CUFILE_SYNC_H

// #include <cuda_runtime.h>
// #include <cufile.h>

// #include "iobench.h"

// namespace mass {

// class CufileSyncIoEngine : public IoEngine {
//  public:
//   CufileSyncIoEngine() = default;
//   ~CufileSyncIoEngine() = default;

//   void Open() override {}
//   void Write(size_t offset, size_t size) override {}
//   void Read(size_t offset, size_t size) override {}
//   void Close() override {}
// };

// void cufile_io(const std::string& filename, size_t transfer_size,
//                size_t block_size, IoPattern io_pattern) {
//   CUfileError_t status = cuFileDriverOpen();
//   if (status.err != CU_FILE_SUCCESS) {
//     return;
//   }
//   CUfileHandle_t fh;
//   CUfileDescr_t params;
//   memset(&params, 0, sizeof(params));
//   int fd = open64(filename.c_str(), O_RDWR | O_CREAT, 0666);
//   if (fd < 0) {
//     int err = errno;
//     std::cerr << "Error opening file: " << filename << " - " << strerror(err)
//               << std::endl;
//     return;
//   }
//   params.handle.fd = fd;
//   params.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
//   status = cuFileHandleRegister(&fh, &params);
//   if (status.err != CU_FILE_SUCCESS) {
//     std::cerr << "cuFileHandleRegister error: " << status.err << std::endl;
//     return;
//   }

//   char* device_buffer;
//   cudaMalloc(&device_buffer, transfer_size);
//   cudaStreamSynchronize(0);
//   status = cuFileBufRegister(device_buffer, transfer_size, 0);
//   if (status.err != CU_FILE_SUCCESS) {
//     std::cerr << "cuFileBufRegister error: " << status.err << std::endl;
//   }

//   size_t num_transfers = block_size / transfer_size;
//   for (size_t i = 0; i < num_transfers; ++i) {
//     size_t offset = i * transfer_size;
//     ssize_t ret = cuFileWrite(fh, device_buffer, transfer_size, offset, 0);
//     if (ret < 0) {
//       std::cerr << "cuFileWrite error: " << ret << std::endl;
//       break;
//     }
//   }
//   cudaFree(device_buffer);
//   cuFileHandleDeregister(fh);
// }

// }  // namespace mass

// #endif


#ifndef CUFILE_SYNC_H
#define CUFILE_SYNC_H

#include <cuda_runtime.h>
#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>
#include <iostream>
#include <string>

#include "iobench.h"

namespace mass {

class CufileSyncIoEngine : public IoEngine {
 public:
  CufileSyncIoEngine() = default;
  ~CufileSyncIoEngine() override {
    Close();
  }

  void Open() override {
    // Initialize cuFile driver
    CUfileError_t status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
      std::cerr << "cuFileDriverOpen error: " << status.err << std::endl;
      return;
    }

    // Open file descriptor
    fd_ = open64(filename_.c_str(), O_RDWR | O_CREAT, 0666);
    if (fd_ < 0) {
      std::cerr << "Error opening file: " << filename_ << " - " << strerror(errno) << std::endl;
      return;
    }

    // Register file handle with libcuFile
    CUfileDescr_t cu_desc;
    std::memset(&cu_desc, 0, sizeof(cu_desc));
    cu_desc.handle.fd = fd_;
    cu_desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    status = cuFileHandleRegister(&fh_, &cu_desc);
    if (status.err != CU_FILE_SUCCESS) {
      std::cerr << "cuFileHandleRegister error: " << status.err << std::endl;
      return;
    }

    // Allocate and register CUDA device buffer
    cudaError_t cuda_status = cudaMalloc(&device_buffer_, transfer_size_);
    if (cuda_status != cudaSuccess) {
      std::cerr << "cudaMalloc error: " << cudaGetErrorString(cuda_status) << std::endl;
      return;
    }
    cudaStreamSynchronize(0);

    status = cuFileBufRegister(device_buffer_, transfer_size_, 0);
    if (status.err != CU_FILE_SUCCESS) {
      std::cerr << "cuFileBufRegister error: " << status.err << std::endl;
    }
  }

  void Write(size_t offset, size_t size) override {
    if (fh_ == nullptr || device_buffer_ == nullptr) return;
    ssize_t ret = cuFileWrite(fh_, device_buffer_, size, offset, 0);
    if (ret < 0) {
      std::cerr << "cuFileWrite error at offset " << offset << ": " << ret << std::endl;
    }
  }

  void Read(size_t offset, size_t size) override {
    if (fh_ == nullptr || device_buffer_ == nullptr) return;
    ssize_t ret = cuFileRead(fh_, device_buffer_, size, offset, 0);
    if (ret < 0) {
      std::cerr << "cuFileRead error at offset " << offset << ": " << ret << std::endl;
    }
  }

  void Close() override {
    if (device_buffer_) {
      cuFileBufDeregister(device_buffer_);
      cudaFree(device_buffer_);
      device_buffer_ = nullptr;
    }
    if (fh_) {
      cuFileHandleDeregister(fh_);
      fh_ = nullptr;
    }
    if (fd_ >= 0) {
      close(fd_);
      fd_ = -1;
    }
    cuFileDriverClose();
  }

 private:
  int fd_{-1};
  CUfileHandle_t fh_{nullptr};
  void* device_buffer_{nullptr};
};

}  // namespace mass

#endif  // CUFILE_SYNC_H
