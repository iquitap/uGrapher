#ifndef GPU_INTRINSICS_H
#define GPU_INTRINSICS_H

#include <iostream>
#include <string>
#include "cublas_v2.h"

#include "infra_gpu/graph.h"
#include "time.h"
#include "infra_gpu/support.h"
#include "infra_gpu/dense.h"
#include "infra_gpu/helper.h"
#include "infra_gpu/gnn_load_balance.h"


#define CUDA_CALL(f) { \
    cudaError_t err = (f); \
    if (err != cudaSuccess) { \
        std::cout \
            << "    Error occurred: " << err << std::endl; \
        std::exit(1); \
    } \
}

#endif
