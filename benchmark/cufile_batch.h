#ifndef CUFILE_BATCH_H
#define CUFILE_BATCH_H

#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cufile.h>

#include "iobench.h"

namespace mass {

class CufileBatchIoEngine : public IoEngine {
 public:
  /// @param filepath   Path to the file to read/write
  /// @param device_id  CUDA device to use
  /// @param io_depth_ Number of I/Os to batch per call
  /// @param transfer_size_    Size (in bytes) of each I/O
  /// @param flags      cuFileBatchIOSubmit flags (default 0)
  explicit CufileBatchIoEngine(const std::string& filepath,
                               int device_id,
                               IoPattern io_pattern,
                               size_t io_depth, // number of I/Os to batch per call
                               size_t transfer_size,
                               size_t block_size,
                               unsigned int flags = 0)
    : filepath_(filepath)
    , device_id_(device_id)
    , flags_(flags)
    , batch_id_(0)
  {
    io_pattern_ = io_pattern;
    transfer_size_ = transfer_size;
    block_size_ = block_size;
    io_depth_ = io_depth;
  }

  ~CufileBatchIoEngine() override {
    // best‐effort cleanup
    try { 
      std::cout << "CufileBatchIoEngine: Destructor" << std::endl;
      Close();
    }
    catch (const std::exception& e) {
      std::cerr << "Exception in CufileBatchIoEngine destructor: " << e.what() << std::endl;
    }
    catch (...) {
      std::cerr << "Unknown exception in CufileBatchIoEngine destructor." << std::endl;
    }
  }

  void Open() override {
    try {
      std::cout << "CufileBatchIoEngine: Opening" << std::endl;
      // select GPU
      if (cudaSetDevice(device_id_) != cudaSuccess) {
        throw std::runtime_error("cudaSetDevice failed");
      }
      std::cout << "CufileBatchIoEngine: cudaSetDevice success" << std::endl;
      // initialize the cuFile driver
      CUfileError_t status = cuFileDriverOpen();
      if (status.err != CU_FILE_SUCCESS) {
        throw std::runtime_error("cuFileDriverOpen failed: " +
                                 std::to_string(status.err));
      }
      std::cout << "CufileBatchIoEngine: cuFileDriverOpen success" << std::endl;
      // resize storage
      fds_.resize(io_depth_, -1);
      cf_descr_.resize(io_depth_);
      cf_handles_.resize(io_depth_);
      dev_ptrs_.resize(io_depth_);
      io_params_.resize(io_depth_);
      io_events_.resize(io_depth_);

      // open & register file handles
      for (size_t i = 0; i < io_depth_; ++i) {
        fds_[i] = open(filepath_.c_str(),
                       O_CREAT | O_RDWR | O_DIRECT,
                       0664);
        if (fds_[i] < 0) {
          throw std::runtime_error("open() failed: " + std::string(std::strerror(errno)));
        }

        std::memset(&cf_descr_[i], 0, sizeof(CUfileDescr_t));
        cf_descr_[i].handle.fd = fds_[i];
        cf_descr_[i].type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

        status = cuFileHandleRegister(&cf_handles_[i], &cf_descr_[i]);
        if (status.err != CU_FILE_SUCCESS) {
          throw std::runtime_error("cuFileHandleRegister failed: " +
                                   std::to_string(status.err));
        }
      }

      // allocate & register GPU buffers
      for (size_t i = 0; i < io_depth_; ++i) {
        if (cudaMalloc(&dev_ptrs_[i], transfer_size_) != cudaSuccess) {
          throw std::runtime_error("cudaMalloc failed");
        }
        // fill with pattern for verification
        cudaMemset(dev_ptrs_[i], 0xAB, transfer_size_);
        cudaStreamSynchronize(0);

        status = cuFileBufRegister(dev_ptrs_[i], transfer_size_, 0);
        if (status.err != CU_FILE_SUCCESS) {
          throw std::runtime_error("cuFileBufRegister failed: " +
                                   std::to_string(status.err));
        }
      }
    } catch (const std::exception& e) {
      std::cerr << "Exception in Open(): " << e.what() << std::endl;
      Close(); // Clean up any partially allocated resources
      throw;   // Rethrow to propagate error
    }
  }

  void Write(size_t offset, size_t size) override {
    if (count_ % io_depth_== io_depth_ - 1){
      submitBatch(offset, size, CUFILE_WRITE);
    }
    count_++;
  }

  void Read(size_t offset, size_t size) override {
    if (count_ % io_depth_== io_depth_ - 1){
      submitBatch(offset, size, CUFILE_READ);
    }
    count_++;
  }

  void Close() override {
    std::cout << "CufileBatchIoEngine: Closing" << std::endl;
    // Deregister & free GPU buffers
    for (size_t i = 0; i < dev_ptrs_.size(); ++i) {
      if (dev_ptrs_[i]) {
        cuFileBufDeregister(dev_ptrs_[i]);
        cudaFree(dev_ptrs_[i]);
        dev_ptrs_[i] = nullptr;
      }
    }
    dev_ptrs_.clear();
    std::cout << "CufileBatchIoEngine: dev_ptrs_.clear success" << std::endl;
    // Deregister file handles & close FDs
    for (size_t i = 0; i < cf_handles_.size(); ++i) {
      if (cf_handles_[i]) {
        cuFileHandleDeregister(cf_handles_[i]);
        cf_handles_[i] = 0;
      }
    }
    cf_handles_.clear();
    std::cout << "CufileBatchIoEngine: cf_handles_.clear success" << std::endl;
    for (size_t i = 0; i < fds_.size(); ++i) {
      if (fds_[i] >= 0) {
        close(fds_[i]);
        fds_[i] = -1;
      }
    }
    fds_.clear();
    std::cout << "CufileBatchIoEngine: fds_.clear success" << std::endl;
    // Shut down driver
    cuFileDriverClose();
    std::cout << "CufileBatchIoEngine: cuFileDriverClose success" << std::endl;
  }

 private:
  void submitBatch(size_t offset,
                   size_t size,
                   CUfileOpcode_t op) {
    // prepare batch parameters
    size_t count = io_depth_;

    for (size_t i = 0; i < count; ++i) {
      
      io_params_[i].mode                  = CUFILE_BATCH;
      io_params_[i].fh                    = cf_handles_[i];
      io_params_[i].u.batch.devPtr_base   = dev_ptrs_[i];
      io_params_[i].u.batch.devPtr_offset = 0;
      io_params_[i].u.batch.file_offset   = offset + i * size;
      io_params_[i].u.batch.size          = size;
      io_params_[i].opcode                = op;

      std::cout << "Setting up IO parameters for batch " << i << ":" << std::endl;
      std::cout << "  Mode: " << (io_params_[i].mode == CUFILE_BATCH ? "CUFILE_BATCH" : "Unknown") << std::endl;
      std::cout << "  File Handle: " << io_params_[i].fh << std::endl;
      std::cout << "  Device Pointer Base: " << io_params_[i].u.batch.devPtr_base << std::endl;
      std::cout << "  Device Pointer Offset: " << io_params_[i].u.batch.devPtr_offset << std::endl;
      std::cout << "  File Offset: " << io_params_[i].u.batch.file_offset << std::endl;
      std::cout << "  Size: " << io_params_[i].u.batch.size << std::endl;
      std::cout << "  Opcode: " << (io_params_[i].opcode == CUFILE_WRITE ? "CUFILE_WRITE" : "CUFILE_READ") << std::endl;
      std::cout << "----------------------------------------" << std::endl;
    }
    
    CUfileError_t err = cuFileBatchIOSetUp(&batch_id_, io_depth_);
    if (err.err != CU_FILE_SUCCESS) {
      throw std::runtime_error("cuFileBatchIOSetUp failed: " +
                               std::to_string(err.err));
    }

    err = cuFileBatchIOSubmit(batch_id_,
                              io_depth_,
                              io_params_.data(),
                              flags_);
    if (err.err != CU_FILE_SUCCESS) {
      cuFileBatchIODestroy(batch_id_);
      throw std::runtime_error("cuFileBatchIOSubmit failed: " +
                               std::to_string(err.err));
    }

    // poll until all completions arrive
    size_t completed = 0;
    while (completed < io_depth_) {
      unsigned int nr = io_depth_;
      err = cuFileBatchIOGetStatus(batch_id_,
                                   io_depth_,
                                   &nr,
                                   io_events_.data(),
                                   nullptr);
      if (err.err != CU_FILE_SUCCESS) {
        cuFileBatchIODestroy(batch_id_);
        throw std::runtime_error("cuFileBatchIOGetStatus failed: " +
                                 std::to_string(err.err));
      }
      completed += nr;
    }

    cuFileBatchIODestroy(batch_id_);
  }

  std::string               filepath_;
  int                       device_id_;
  unsigned int              flags_;
  CUfileBatchHandle_t       batch_id_;
  size_t                    count_ = 0;

  std::vector<int>                fds_;
  std::vector<CUfileDescr_t>      cf_descr_;
  std::vector<CUfileHandle_t>     cf_handles_;
  std::vector<void*>              dev_ptrs_;
  std::vector<CUfileIOParams_t>   io_params_;
  std::vector<CUfileIOEvents_t>   io_events_;
};

}  // namespace mass

#endif  // CUFILE_BATCH_H
