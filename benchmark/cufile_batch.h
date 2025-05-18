#ifndef CUFILE_BATCH_H
#define CUFILE_BATCH_H

#include <cuda_runtime.h>
#include <cufile.h>

#include "iobench.h"

namespace mass {

class CufileBatchIoEngine : public IoEngine {
 public:
  CufileBatchIoEngine() = default;
  ~CufileBatchIoEngine() = default;

  void Open() override {}
  void Write(size_t offset, size_t size) override {}
  void Read(size_t offset, size_t size) override {}
  void Close() override {}
};

}  // namespace mass

#endif  // CUFILE_BATCH_H
