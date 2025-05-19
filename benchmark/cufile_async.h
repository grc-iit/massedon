#ifndef CUFILE_ASYNC_H
#define CUFILE_ASYNC_H

#include <cuda_runtime.h>
#include <cufile.h>
#include <fcntl.h> // For open, O_RDWR, etc.
#include <unistd.h> // For close
#include <sys/types.h> // For off_t
#include <sys/stat.h>
#include <cstring> // For memset
#include <iostream>
#include <vector>
#include <stdexcept>
#include "iobench.h"

namespace mass {

class CufileAsyncIoEngine : public IoEngine {
  struct io_args_s
  {
    void *devicePtr = nullptr;
    off_t offset ;
    off_t buf_off;
    ssize_t bytes_done = 0;
    size_t last_io_size = 0;
    size_t data_size = 0;
  };

  private:
  int fd_ = -1;
  CUfileError_t status;
  CUfileDescr_t cf_descr;
  CUfileHandle_t cf_handle = nullptr;
  size_t total_size;
  std::vector<io_args_s> args;
  int io_depth_;
  size_t count_ = 0;
  // io stream associated with the I/O
	cudaStream_t io_stream;


  public:
  CufileAsyncIoEngine(size_t transfer_size, size_t block_size, IoPattern io_pattern, float percent_read, const std::string& filename, int io_depth) 
    : IoEngine() {
    
    this->transfer_size_ = transfer_size;
    this->block_size_ = block_size;
    this->io_pattern_ = io_pattern;
    this->percent_read_ = static_cast<int>(percent_read); // Cast float to int
    this->filename_ = filename;
    this->io_depth_ = io_depth;
    total_size = transfer_size * io_depth;
    args.resize(io_depth_);
  }

  ~CufileAsyncIoEngine(){
    Close();
  }
  void Open() override;
  void Write(size_t offset, size_t size) override;
  void Read(size_t offset, size_t size) override;
  void Close() override ;
  void Verify();
};

void CufileAsyncIoEngine::Open() {
  std::cout << "Opening CufileAsync" << std::endl;

  status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "cuFileDriverOpen failed: " << status.err << std::endl;
        return ;
    }

  cudaSetDevice(0);

  // allocate device memory and create streams
  for (int i = 0; i < io_depth_; i++) {
    std::cout << "Allocating device memory for io : " << i << std::endl;
    if (cudaMalloc(&args[i].devicePtr, transfer_size_) != cudaSuccess) {
      std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return ;
    }
    status = cuFileBufRegister(args[i].devicePtr, transfer_size_, 0);
    if (status.err != CU_FILE_SUCCESS) {
      std::cerr << "cuFileBufRegister failed: " << status.err << std::endl;
      return ;
    }
    args[i].buf_off = 0;                 // always 0 unless using sub-buffers
    args[i].data_size = transfer_size_;
    args[i].bytes_done = 0;
    args[i].last_io_size = 0;
    cudaStreamCreateWithFlags(&io_stream, cudaStreamNonBlocking);
  }


  fd_ = open(filename_.c_str(), O_RDWR | O_CREAT | O_DIRECT, 0644);
  if (fd_ == -1){
    std::cerr << "Failed to open file: " << filename_ << std::endl;
    return ;
  }
  else{
    std::cout << "File opened successfully" << std::endl;
  }

  // register file handle
  memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
  cf_descr.handle.fd = fd_;
  cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  status = cuFileHandleRegister(&cf_handle, &cf_descr);
  if (status.err != CU_FILE_SUCCESS) {
    std::cerr << "cuFileHandleRegister failed: " << status.err << std::endl;
    return ;
  }
  else{
    std::cout << "File handle registered successfully" << std::endl;
  }

}

void CufileAsyncIoEngine::Close(){
  std::cout << "Closing CufileAsync" << std::endl;
  for (int i = 0; i < io_depth_; i++) {
    std::cout << "Closing CufileAsync for io : " << i << std::endl;

    cudaStreamSynchronize(io_stream);

    if (args[i].bytes_done != (ssize_t)transfer_size_) {
      std::cerr << "Read for IO[" << i << "] less than requested: " << args[i].bytes_done << " < " << transfer_size_ << std::endl;
    }

    if (args[i].devicePtr) {
      cuFileBufDeregister(args[i].devicePtr);
      cudaFree(args[i].devicePtr);
      args[i].devicePtr = nullptr;
    }
    if (io_stream) {
      cudaStreamDestroy(io_stream);
      io_stream = nullptr;
    }
  }
  if (cf_handle) {
    cuFileHandleDeregister(cf_handle);
    cf_handle = nullptr;
  }
  if (fd_ != -1) {
    close(fd_);
    fd_ = -1;
  }
}

void CufileAsyncIoEngine::Read(size_t offset, size_t size){
  std::cout << "Reading from CufileAsync" << std::endl;

  int i = count_ % io_depth_;
  args[i].offset = offset;           // file offset for this I/O
  args[i].buf_off = 0;               // always 0 unless using sub-buffers
  status = cuFileReadAsync(cf_handle, args[i].devicePtr, &args[i].data_size, &args[i].offset, &args[i].buf_off, &args[i].bytes_done, io_stream);
  if (status.err != CU_FILE_SUCCESS) {
      std::cerr << "cuFileReadAsync failed: " << status.err << std::endl;
    }
  args[i].last_io_size = args[i].data_size;

  if(i ==  io_depth_ - 1) {
    cudaStreamSynchronize(io_stream);
    Verify();
  }
  count_++;
}

void CufileAsyncIoEngine::Write(size_t offset, size_t size){
  std::cout << "Writing to CufileAsync" << std::endl;

  int i = count_ % io_depth_;
  args[i].offset = offset;            // file offset for this I/O
  args[i].buf_off = 0; 

  status = cuFileWriteAsync(cf_handle, args[i].devicePtr, &args[i].data_size, &args[i].offset, &args[i].buf_off, &args[i].bytes_done, io_stream);

  if (status.err != CU_FILE_SUCCESS) {
      std::cerr << "cuFileWriteAsync failed: " << status.err << std::endl;
    }
  

  if(i ==  io_depth_ - 1) {
      cudaStreamSynchronize(io_stream);
      Verify();
    }
  count_++;
}

void CufileAsyncIoEngine::Verify() {
  std::cout << "Verifying CufileAsync" << std::endl;
  for (int i = 0; i < io_depth_; i++) {
    if (args[i].bytes_done != (ssize_t)args[i].data_size) {
      std::cout << "args[" << i << "]: "
              << "devicePtr=" << args[i].devicePtr << ", "
              << "offset=" << args[i].offset << ", "
              << "buf_off=" << args[i].buf_off << ", "
              << "bytes_done=" << args[i].bytes_done << ", "
              << "last_io_size=" << args[i].last_io_size << ", "
              << "data_size=" << args[i].data_size << std::endl;
      }
  }
}

}  // namespace mass




#endif