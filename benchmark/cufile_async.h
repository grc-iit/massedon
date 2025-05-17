#ifndef CUFILE_ASYNC_H
#define CUFILE_ASYNC_H

#include <cuda_runtime.h>
#include <cufile.h>

#include "iobench.h"

namespace mass {

class CufileAsyncIoEngine : public IoEngine {
 public:
  CufileAsyncIoEngine() = default;
  ~CufileAsyncIoEngine() = default;
  void Open() override {}
  void Write(size_t offset, size_t size) override {}
  void Read(size_t offset, size_t size) override {}
  void Close() override {}
};

}  // namespace mass

#endif