#include <cuda_runtime.h>
#include <fcntl.h>
#include <hermes_shm/util/config_parse.h>
#include <hermes_shm/util/timer_mpi.h>
#include <mpi.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "iobench_factory"

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (argc != 7) {
    if (rank == 0) {
      std::cerr
          << "Usage: " << argv[0]
          << " <transfer_size> <block_size> <io_pattern (random|sequential)>"
          << " <io_type (read|write)> <io_engine (posix|cufile)> <filename>"
          << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  size_t transfer_size = hshm::ConfigParse::ParseSize(argv[1]);
  size_t block_size = hshm::ConfigParse::ParseSize(argv[2]);
  std::string io_pattern_str = argv[3];
  std::string io_type_str = argv[4];
  std::string io_engine_str = argv[5];
  std::string filename = argv[6];

  IOPattern io_pattern;
  if (io_pattern_str == "random") {
    io_pattern = IOPattern::RANDOM;
  } else if (io_pattern_str == "sequential") {
    io_pattern = IOPattern::SEQUENTIAL;
  } else {
    if (rank == 0) {
      std::cerr << "Invalid IO pattern: " << io_pattern_str << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  IOType io_type;
  if (io_type_str == "read") {
    io_type = IOType::READ;
  } else if (io_type_str == "write") {
    io_type = IOType::WRITE;
  } else {
    if (rank == 0) {
      std::cerr << "Invalid IO type: " << io_type_str << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  IOEngine io_engine;
  if (io_engine_str == "posix") {
    io_engine = IOEngine::POSIX;
  }
#ifdef USE_CUFILE
  else if (io_engine_str == "cufile") {
    io_engine = IOEngine::CUFILE;
  }
#endif
  else {
    if (rank == 0) {
      std::cerr << "Invalid IO engine: " << io_engine_str << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  hshm::MpiTimer timer(MPI_COMM_WORLD);
  timer.Resume();
  switch (io_engine) {
    case IOEngine::POSIX:
      break;
#ifdef US_CUFILE
    case IOEngine::CUFILE:
      break;
#endif
  }
  timer.Pause();
  timer.Collect();  // Calls MPI_Barrier

  if (rank == 0) {
    double duration = timer.GetMsec();
    double bandwidth =
        static_cast<double>(block_size) / (average_duration / 1000.0);
    HILOG(kInfo,
          "IoBench done in: api={} pattern={} io_type={} nprocs={} time={}ms "
          "io_size={}bytes",
          io_engine_str, io_pattern_str, io_type_str, nprocs, average_duration,
          block_size * nprocs);
  }

  MPI_Finalize();
  return 0;
}
