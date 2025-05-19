#include "iobench.h"

#include <hermes_shm/util/config_parse.h>
#include <hermes_shm/util/timer_mpi.h>
#include <mpi.h>

#include <numeric>
#include <random>
#include <vector>

#include "iobench_factory.h"

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (argc != 8) {
    if (rank == 0) {
      std::cerr
          << "Usage: " << argv[0]
          << " <transfer_size> <block_size> <io_pattern (random|sequential)>"
          << " <percent_read (0-100)> <io_engine (posix|cufile)> <filename> <io_depth>"
          << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  size_t transfer_size = hshm::ConfigParse::ParseSize(argv[1]);
  size_t block_size = hshm::ConfigParse::ParseSize(argv[2]);
  std::string io_pattern_str = argv[3];
  float percent_read = hshm::ConfigParse::ParseNumber<float>(argv[4]);
  std::string io_engine_str = argv[5];
  std::string filename = argv[6];
  // adding for io_depth
  int io_depth = atoi(argv[7]);

  std::cout << "IO depth: " << io_depth << std::endl;

  mass::IoPattern io_pattern = mass::IoPatternFactory::Get(io_pattern_str);

  std::unique_ptr<mass::IoEngine> io_engine =
      mass::IoEngineFactory::Get(io_engine_str, transfer_size, block_size, io_pattern, percent_read, filename, io_depth); 

  MPI_Barrier(MPI_COMM_WORLD);
  hshm::MpiTimer timer(MPI_COMM_WORLD);
  timer.Resume();
  std::cout << "Running IO engine" << std::endl;
  io_engine->Run();
  timer.Pause();
  timer.Collect();  // Calls MPI_Barrier

  if (rank == 0) {
    double duration = timer.GetMsec();
    double bandwidth = static_cast<double>(block_size) / (duration / 1000.0);
    HILOG(kInfo,
          "IoEngine done in: api={} pattern={} percent_read={} nprocs={} "
          "time={}ms "
          "io_size={}bytes",
          io_engine_str, io_pattern_str, percent_read, nprocs, duration,
          block_size * nprocs);
  }

  MPI_Finalize();
  return 0;
}
