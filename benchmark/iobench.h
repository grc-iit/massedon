#ifndef IOBENCH_H
#define IOBENCH_H

#include <hermes_shm/util/random.h>
#include <mpi.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

namespace mass {

enum class IoPattern { kSequential, kRandom };

class IoEngine {
 public:
  std::string filename_;
  size_t transfer_size_;  // Data size in each read / write call
  size_t block_size_;  // Size of data per process (i.e., larger than transfer
                       // size)
  IoPattern io_pattern_;
  int rank_;          // MPI rank
  int nprocs_;        // MPI # processes
  int percent_read_;  // Percent chance of read
  hshm::UniformDistribution is_read_distrib_;

 public:
  IoEngine() {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs_);
    is_read_distrib_.Seed(rank_ + 8124891849);
    is_read_distrib_.Shape(0, 100);
  }
  virtual void Open();
  virtual void Write(size_t offset, size_t size);
  virtual void Read(size_t offset, size_t size);
  virtual void Close();

  void Run() {
    switch (io_pattern_) {
      case IoPattern::kSequential: {
        SequentialIo();
        break;
      }
      case IoPattern::kRandom: {
        RandomIo();
        break;
      }
    }
  }

  bool CheckIfRead() {
    if (percent_read_ == 0) {
      return false;
    }
    if (percent_read_ == 100) {
      return true;
    }
    float test = is_read_distrib_.GetDouble();
    return test <= percent_read_;
  }

  void SequentialIo() {
    Open();
    size_t proc_off = rank_ * block_size_;
    for (size_t i = 0; i < block_size_; i += transfer_size_) {
      size_t offset = i + proc_off;
      bool is_read = CheckIfRead();
      if (is_read) {
        Read(offset, transfer_size_);
      } else {
        Write(offset, transfer_size_);
      }
    }
    Close();
  }

  void RandomIo() {
    Open();
    hshm::UniformDistribution distrib;
    distrib.Seed(rank_ + 8124891849);
    distrib.Shape(0, block_size_ - transfer_size_);
    for (size_t i = 0; i < block_size_; i += transfer_size_) {
      size_t offset = distrib.GetSize();
      bool is_read = CheckIfRead();
      if (is_read) {
        Read(offset, transfer_size_);
      } else {
        Write(offset, transfer_size_);
      }
    }
    Close();
  }
};

}  // namespace mass

#endif