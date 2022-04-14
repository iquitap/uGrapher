// Copyright (c) 2022, Yangjie Zhou.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
   
#ifndef DENSE_H
#define DENSE_H

#include "infra_gpu/graph.h"
#include <cooperative_groups.h>

#include "cublas_v2.h"
using namespace cooperative_groups;


namespace gpu_runtime {

    void matmul_NN(cublasHandle_t *cublasHs, float* d_A, float* d_B, float* d_C, int inM, int inN, int inK, float alpha = 1, float beta =  0)
    {
        assert(cublasHs[0] != NULL);
        // const float alpha = 1;
        // const float beta = 0;

        checkCudaErrors(cublasSgemm(
            cublasHs[0],
            CUBLAS_OP_N,    
            CUBLAS_OP_N,    
            inN,
            inM,
            inK,
            &alpha,         
            d_B,            
            inN,            
            d_A,            
            inK,            
            &beta,         
            d_C,            
            inN             
        ));
    
    }

}

#endif 