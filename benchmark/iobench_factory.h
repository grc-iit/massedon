#ifndef IOBENCH_RUNTIME_H
#define IOBENCH_RUNTIME_H

#include <memory>

#include "cufile_async.h"
#include "cufile_sync.h"
#include "cufile_batch.h"
#include "iobench.h"
#include "posix.h"

namespace mass {

class IoPatternFactory {
 public:
  static IoPattern Get(const std::string &io_pattern_str) {
    if (io_pattern_str == "random") {
      return IoPattern::kRandom;
    } else if (io_pattern_str == "sequential") {
      return IoPattern::kSequential;
    } else {
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank == 0) {
        std::cerr << "Invalid IO pattern: " << io_pattern_str << std::endl;
      }
      MPI_Finalize();
      exit(1);
    }
  }
};

class IoEngineFactory {
 public:
  template <typename... Args>
  static std::unique_ptr<IoEngine> Get(const std::string &io_engine_str,
                                       Args &&...args) {
    if (io_engine_str == "posix") {
      return std::make_unique<PosixIoEngine>();
    } else if (io_engine_str == "cufile") {
      return std::make_unique<CufileSyncIoEngine>();
    } else if (io_engine_str == "cufile_async") {
      return std::make_unique<CufileAsyncIoEngine>();
    } else if (io_engine_str == "cufile_batch") {
      return std::make_unique<CufileBatchIoEngine>();
    } else {
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank == 0) {
        std::cerr << "Invalid IO engine: " << io_engine_str << std::endl;
      }
      MPI_Finalize();
      exit(1);
    }
  }
};

}  // namespace mass

#endif