#ifndef IOBENCH_POSIX_H
#define IOBENCH_POSIX_H

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <system_error>

#include "iobench.h"

#define CUDA_CHECK(call)                                                        \
  do {                                                                           \
    cudaError_t err__ = (call);                                                  \
    if (err__ != cudaSuccess) {                                                  \
      throw std::system_error(                                                   \
        static_cast<int>(err__),                                                \
        std::generic_category(),                                                 \
        "CUDA operation failed: " + std::string(cudaGetErrorString(err__)));        \
    }                                                                            \
  } while (0)

namespace mass {

class PosixIoEngine : public IoEngine {
 private:
  int fd_;
  char* host_buffer_;
  char* device_buffer_;

 public:
  // Constructor with parameters
  PosixIoEngine(size_t transfer_size, size_t block_size, IoPattern io_pattern, float percent_read, const std::string& filename) 
    : IoEngine(), // Call base class default constructor
      fd_(-1), host_buffer_(nullptr), device_buffer_(nullptr) {
    // Assign to public members of the base class
    this->transfer_size_ = transfer_size;
    this->block_size_ = block_size;
    this->io_pattern_ = io_pattern;
    this->percent_read_ = static_cast<int>(percent_read); // Cast float to int
    this->filename_ = filename;
  }
  ~PosixIoEngine() {
    Close();
  }

  void Open() override;
  void Write(size_t offset, size_t size) override;
  void Read(size_t offset, size_t size) override;
  void Close() override;
};

void PosixIoEngine::Open() {
  std::cout << "Opening POSIX" << std::endl;

  std::cout << "Allocating host memory" << std::endl;
  // allocate mem in host buffer
  if(cudaMallocHost(&host_buffer_, transfer_size_) != cudaSuccess) {
    throw std::runtime_error("Failed to allocate host memory");
  }

  // allocate mem in GPU
  if (cudaMalloc(&device_buffer_, transfer_size_) != cudaSuccess) {
    cudaFreeHost(host_buffer_); // if fails, free the host mem
    throw std::runtime_error("Failed to allocate device memory");
  }

  std::cout << "Opening file" << std::endl;
  // open file
  fd_ = open(filename_.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_SYNC | O_DIRECT, 0644);
  if (fd_ == -1) {
    // free the mem allocated in host and device
    cudaFreeHost(host_buffer_);
    cudaFree(device_buffer_);
    throw std::runtime_error("Failed to open file: " + filename_);
  }
  else{
    std::cout << "File opened successfully" << std::endl;
  }
}

void PosixIoEngine::Close() {
  std::cout << "Closing POSIX" << std::endl;
  if (fd_ != -1) {
      close(fd_);
      fd_ = -1;
    }
    if (host_buffer_) {
      cudaFreeHost(host_buffer_);
      host_buffer_ = nullptr;
    }
    if (device_buffer_) {
      cudaFree(device_buffer_);
      device_buffer_ = nullptr;
    }
}

void PosixIoEngine::Read(size_t offset, size_t size) {
  std::cout << "Reading from POSIX" << std::endl;
  if (fd_ == -1 || !host_buffer_ || !device_buffer_) {
    throw std::runtime_error("File not opened or buffers not initialized");
  }

  if (lseek(fd_, offset, SEEK_SET) == -1) {
    throw std::system_error(errno, std::generic_category(), 
      "Error seeking to offset: " + std::to_string(offset));
  }

  std::cout << "Copying from device to host" << std::endl;
  // First copy from device to host buffer
  CUDA_CHECK(cudaMemcpy(host_buffer_, device_buffer_, size, cudaMemcpyDeviceToHost));

  // Then write from host buffer to file
  ssize_t bytes_written = write(fd_, host_buffer_, size);

  if (bytes_written == -1) {
    throw std::system_error(errno, std::generic_category(),
      "Error writing to file at offset: " + std::to_string(offset));
  }
  if (static_cast<size_t>(bytes_written) != size) {
    throw std::runtime_error("Wrote less than requested: " + 
      std::to_string(bytes_written) + " < " + std::to_string(size));
  }
}

void PosixIoEngine::Write(size_t offset, size_t size) {
  std::cout << "Writing to POSIX" << std::endl;
  if (fd_ == -1 || !host_buffer_ || !device_buffer_) {
    throw std::runtime_error("File not opened or buffers not initialized");
  }
  
  if (lseek(fd_, offset, SEEK_SET) == -1) {
    throw std::system_error(errno, std::generic_category(),
      "Error seeking to offset: " + std::to_string(offset));
  }

  std::cout << "Writing to file" << std::endl;
  ssize_t written = write(fd_, host_buffer_, size);
  std::cout << "Written: " << written << std::endl;
  
  std::cout << "Checking if write was successful" << std::endl;
  if (written == -1) {
    throw std::system_error(errno, std::generic_category(),
      "Error writing to file at offset: " + std::to_string(offset));
  }
  if (static_cast<size_t>(written) != size) {
    throw std::runtime_error("Write less than requested: " + 
      std::to_string(written) + " < " + std::to_string(size));
  }

  std::cout << "Copying from host to device" << std::endl;
  CUDA_CHECK(cudaMemcpy(device_buffer_, host_buffer_, size, cudaMemcpyHostToDevice));
}

void posix_io(const std::string& filename, size_t transfer_size,
              size_t block_size, IoPattern io_pattern) {
  char* host_buffer_;
  char* device_buffer_;

  cudaMallocHost(&host_buffer_, transfer_size);
  cudaMalloc(&device_buffer_, transfer_size);
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error opening file for writing." << std::endl;
    cudaFreeHost(host_buffer_);
    cudaFree(device_buffer_);
    return;
  }

  std::vector<char> buffer(transfer_size);

  size_t num_transfers = block_size / transfer_size;
  for (size_t i = 0; i < num_transfers; ++i) {
    size_t offset = i * transfer_size;
    file.seekp(offset);
    file.write(buffer.data(), transfer_size);
    cudaMemcpy(device_buffer_, buffer.data(), transfer_size,
               cudaMemcpyHostToDevice);
  }
  file.close();
  cudaFreeHost(host_buffer_);
  cudaFree(device_buffer_);
}

}  // namespace mass

#endif