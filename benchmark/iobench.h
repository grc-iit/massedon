#ifndef IOBENCH_H
#define IOBENCH_H

#include <mpi.h>

enum class IOPattern { SEQUENTIAL, RANDOM };

enum class IOType { READ, WRITE };

enum class IOEngine { POSIX, CUFILE };

class IoBench {
 public:
  std::string filename_;
  size_t transfer_size_;  // Data size in each read / write call
  size_t block_size_;  // Size of data per process (i.e., larger than transfer
                       // size)
  IOPattern io_pattern_;
  int rank_;          // MPI rank
  int nprocs_;        // MPI # processes
  int percent_read_;  // Percent chance of read
  std::random_device is_read_dev_;
  std::mt19937 is_read_gen_;
  std::uniform_int_distribution<size_t> is_read_distrib_;

 public:
  IoBench() {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs_);
    is_read_gen_ = std::mt19937 gen(is_read_dev_());
    is_read_distrib_ = std::uniform_int_distribution<float>(0, 100);
  }
  virtual void Open();
  virtual void Write(size_t offset, size_t size);
  virtual void Read(size_t offset, size_t size);
  virtual void Close();

  bool CheckIfRead() {
    if (percent_read_ == 0) {
      return false;
    }
    if (percent_read_ == 100) {
      return true;
    }
    float test = is_read_distrib_(gen);
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
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> distrib(0,
                                                  block_size - transfer_size);
    for (size_t i = 0; i < 0; ++i) {
      size_t offset = distrib(gen);
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

#endif