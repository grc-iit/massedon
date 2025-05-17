#ifndef IOBENCH_POSIX_H
#define IOBENCH_POSIX_H

#include <fcntl.h>
#include <unistd.h>

#include "iobench.h"

namespace mass {

// class PosixIoEngine : public IoEngine {
//  public:
// };

void posix_io(const std::string& filename, size_t transfer_size,
              size_t block_size, IOType io_type, IoPattern io_pattern) {
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
    size_t offset =
        (io_pattern == IoPattern::kRandom) ? distrib(gen) : i * transfer_size;
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