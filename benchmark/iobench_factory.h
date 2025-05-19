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
                                       size_t transfer_size, size_t block_size, IoPattern io_pattern, float percent_read, const std::string& filename, int io_depth) {

    std::cout << "IoEngineFactory: checking for: " << io_engine_str << std::endl;
    if (io_engine_str == "posix") {
      return std::make_unique<PosixIoEngine>(transfer_size, block_size, io_pattern, percent_read, filename);
    } else if (io_engine_str == "cufile") {
      return std::make_unique<CufileSyncIoEngine>(transfer_size, block_size, io_pattern, percent_read, filename);
    } else if (io_engine_str == "cufile_async") {
      return std::make_unique<CufileAsyncIoEngine>(transfer_size, block_size, io_pattern, percent_read, filename, io_depth);
    } else if (io_engine_str == "cufile_batch") {
      return std::make_unique<CufileBatchIoEngine>(filename, 0, io_depth, transfer_size, 0);
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