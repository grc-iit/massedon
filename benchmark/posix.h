#ifndef IOBENCH_POSIX_H
#define IOBENCH_POSIX_H

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "iobench.h"

namespace mass {

class PosixIoEngine : public IoEngine {
 public:
  PosixIoEngine() = default;
  ~PosixIoEngine() = default;

  void Open() override {}
  void Write(size_t offset, size_t size) override {}
  void Read(size_t offset, size_t size) override {}
  void Close() override {}
};

void posix_io(const std::string& filename, size_t transfer_size,
              size_t block_size, IoPattern io_pattern) {
  char* host_buffer;
  char* device_buffer;

  cudaMallocHost(&host_buffer, transfer_size);
  cudaMalloc(&device_buffer, transfer_size);
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error opening file for writing." << std::endl;
    cudaFreeHost(host_buffer);
    cudaFree(device_buffer);
    return;
  }

  std::vector<char> buffer(transfer_size);

  size_t num_transfers = block_size / transfer_size;
  for (size_t i = 0; i < num_transfers; ++i) {
    size_t offset = i * transfer_size;
    file.seekp(offset);
    file.write(buffer.data(), transfer_size);
    cudaMemcpy(device_buffer, buffer.data(), transfer_size,
               cudaMemcpyHostToDevice);
  }
  file.close();
  cudaFreeHost(host_buffer);
  cudaFree(device_buffer);
}

}  // namespace mass

#endif