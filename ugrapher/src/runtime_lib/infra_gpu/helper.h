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

#ifndef HELPER_H
#define HELPER_H

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}


void write_array_into_txt(float *array, int size, char *filename){
    FILE *fp = fopen(filename, "w");
    for(int i = 0; i < size; i++){
        fprintf(fp, "%lf\n", array[i]);
    }
    fclose(fp);    
}

void write_device_array_into_txt(float *array, int size, char *filename){
    float *host_array = (float *)malloc(sizeof(float) * size);
    cudaMemcpy(host_array, array, sizeof(float) * size, cudaMemcpyDeviceToHost);

    write_array_into_txt(host_array, size, filename);   
}

void print_matrix_into_txt(float *matrix, int M, int N, char *filename){
    FILE *fp = fopen(filename, "w");
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            fprintf(fp, "%lf ", matrix[i * N + j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void write_device_matrix_into_txt(float *matrix, int M, int N, char *filename){
    float *host_matrix = (float *)malloc(sizeof(float) * M * N);
    cudaMemcpy(host_matrix, matrix, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    print_matrix_into_txt(host_matrix, M, N, filename);
}

void init_host_matrix_rand(float *matrix, int M, int N){
    printf("Initializing host matrix with random values...\n");
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            matrix[i * N + j] = (float)rand() / (float)RAND_MAX;
        }
    }
    printf("Matrix initialized.\n");
}

void init_host_matrix_static(float *matrix, int M, int N){
    printf("Initializing matrix with static values...\n");
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            matrix[i * N + j] = 1.0;
        }
    }
    printf("Matrix initialized\n");
}

__global__ void initGPUData_ker(float *data, int numElements, float value) {
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   if (tid < numElements) {
      data[tid] = value;
   }
}

void initGPUData(float *data, int numElements, float value) {
   dim3 gridDim;
   dim3 blockDim;

   blockDim.x = 1024;
   gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;

   initGPUData_ker <<< gridDim, blockDim >>> (data, numElements, value);
}


__global__ void concat_matrix_ker(float *d_A, float *d_B, float *d_C, int M, int N1, int N2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < M * (N1 + N2)) {
        int i = tid / (N1 + N2);
        int j = tid % (N1 + N2);
        if (j < N1) {
            d_C[tid] = d_A[i * N1 + j];
        } else {
            d_C[tid] = d_B[i * N2 + j - N1];
        }
    }
}

// concat two matrix
// d_a: [M, N1]
// d_b: [M, N2]
// d_c: [M, N1 + N2]
void concat_matrix(float* d_A, float* d_B, float* d_C, int M, int N1, int N2){
    dim3 gridDim;
    dim3 blockDim;

    //thread_num = M*(N1+N2)
    //each thread copy one element
    blockDim.x = 1024;
    gridDim.x = (M * (N1 + N2) + blockDim.x - 1) / blockDim.x;

    concat_matrix_ker <<< gridDim, blockDim >>> (d_A, d_B, d_C, M, N1, N2);
}

__global__ void split_matrix_ker (float *d_A, float *d_B, float *d_C, int M, int N1, int N2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < M * (N1 + N2)) {
        int i = tid / (N1 + N2);
        int j = tid % (N1 + N2);
        if (j < N1) {
            d_A[i * N1 + j] = d_C[tid];
        } else {
            d_B[i * N2 + j - N1] = d_C[tid];
        }
    }
}
// split matrix into two matrix
// d_a: [M, N1]
// d_b: [M, N2]
// d_c: [M, N1 + N2]
void split_matrix(float *d_A, float *d_B, float *d_C, int M, int N1, int N2){
    dim3 gridDim;
    dim3 blockDim;

    //thread_num = M*(N1+N2)
    //each thread copy one element
    blockDim.x = 1024;
    gridDim.x = (M * (N1 + N2) + blockDim.x - 1) / blockDim.x;

    split_matrix_ker <<< gridDim, blockDim >>> (d_A, d_B, d_C, M, N1, N2);
}






//use memcpy to implement concat_matrix
// input matrix d_A: [M, N1]
// input matrix d_B: [M, N2]
// output matrix d_C: [M, N1 + N2]
void concat_matrix_memcpy(float* d_A, float* d_B, float* d_C, int M, int N1, int N2){
    for(int i = 0; i < M; i++){
        cudaMemcpy(&d_C[i * (N1 + N2)], &d_A[i * N1], sizeof(float) * N1, cudaMemcpyDeviceToDevice);
        cudaMemcpy(&d_C[i * (N1 + N2) + N1], &d_B[i * N2], sizeof(float) * N2, cudaMemcpyDeviceToDevice);
    }
}


// use memcpy to implicit split_matrix
// input matrix d_C: [M, N1 + N2]
// output matrix d_A: [M, N1]
// output matrix d_B: [M, N2]
void split_matrix_memcpy(float* d_A, float* d_B, float* d_C, int M, int N1, int N2){
    for(int i = 0; i < M; i++){
        cudaMemcpy(&d_A[i * N1], &d_C[i * (N1 + N2)], sizeof(float) * N1, cudaMemcpyDeviceToDevice);
        cudaMemcpy(&d_B[i * N2], &d_C[i * (N1 + N2) + N1], sizeof(float) * N2, cudaMemcpyDeviceToDevice);
    }
}

// use memcpyAsyc to implement concat_matrix
// input matrix d_A: [M, N1]
// input matrix d_B: [M, N2]
// output matrix d_C: [M, N1 + N2]
void concat_matrix_memcpyAsync(float* d_A, float* d_B, float* d_C, int M, int N1, int N2){
    for(int i = 0; i < M; i++){
        cudaMemcpyAsync(&d_C[i * (N1 + N2)], &d_A[i * N1], sizeof(float) * N1, cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(&d_C[i * (N1 + N2) + N1], &d_B[i * N2], sizeof(float) * N2, cudaMemcpyDeviceToDevice);
    }
}


static inline __device__ void MyatomicMax(float *addr, float val) {
    if (*addr >= val) return;

    unsigned int *const addr_as_ui = (unsigned int *)addr;
    unsigned int old = *addr_as_ui, assumed;
    do {
        assumed = old;
        if (__uint_as_float(assumed) >= val) break;
        old = atomicCAS(addr_as_ui, assumed, __float_as_uint(val));
    } while (assumed != old);
}


#endif // HELPER_H