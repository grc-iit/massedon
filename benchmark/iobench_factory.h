#ifdef IOBENCH_RUNTIME_H
#define IOBENCH_RUNTIME_H

#define USE_CUFILE
#ifdef USE_CUFILE
#include "cufile_async.h"
#include "cufile_sync.h"
#endif
#include "posix.h"

#endif