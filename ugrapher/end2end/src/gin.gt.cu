#include "gpu_intrinsics.h"
#include <cooperative_groups.h>

// total feature size
int32_t sz_fin;		    // (model_fin  * model_vertex)
int32_t sz_fhid;		// (model_fhid * model_vertex)
int32_t sz_fout;		// (model_fout * model_vertex)

int32_t sz_w1;          // (model_fin * model_fhid)
int32_t sz_w2;          // (model_fhid * model_fhid)
int32_t sz_w3;          // (model_fhid * model_fout)

#define eps                 0.0

using namespace cooperative_groups;
int32_t __delta_param;
gpu_runtime::GraphT __device__ edges;
gpu_runtime::GraphT __host_edges;
gpu_runtime::GraphT __device__ edges__transposed;
gpu_runtime::GraphT __host_edges__transposed;

int32_t __device__ M; 
int32_t __host_M;

int32_t __device__ input_f_size; 
int32_t __host_input_f_size;

int32_t __device__ hidden_f_size; 
int32_t __host_hidden_f_size;

int32_t __device__ output_f_size; 
int32_t __host_output_f_size;

int32_t __device__ vector_size; 
int32_t __host_vector_size;

float __device__ *feature_input;
float *__host_feature_input;
float *__device_feature_input;

float __device__ *feature_input_out_1;
float *__host_feature_input_out_1;
float *__device_feature_input_out_1;

float __device__ *feature_input_eps_1;
float *__host_feature_input_eps_1;
float *__device_feature_input_eps_1;

float __device__ *feature_rst_1;
float *__host_feature_rst_1;
float *__device_feature_rst_1;

float __device__ *weight_1;
float *__host_weight_1;
float *__device_weight_1;

float __device__ *feature_input_out_2;
float *__host_feature_input_out_2;
float *__device_feature_input_out_2;

float __device__ *feature_input_eps_2;
float *__host_feature_input_eps_2;
float *__device_feature_input_eps_2;

float __device__ *feature_rst_2;
float *__host_feature_rst_2;
float *__device_feature_rst_2;

float __device__ *weight_2;
float *__host_weight_2;
float *__device_weight_2;

float __device__ *feature_input_out_3;
float *__host_feature_input_out_3;
float *__device_feature_input_out_3;

float __device__ *feature_input_eps_3;
float *__host_feature_input_eps_3;
float *__device_feature_input_eps_3;

float __device__ *feature_rst_3;
float *__host_feature_rst_3;
float *__device_feature_rst_3;

float __device__ *weight_3;
float *__host_weight_3;
float *__device_weight_3;

float __device__ *feature_input_out_4;
float *__host_feature_input_out_4;
float *__device_feature_input_out_4;

float __device__ *feature_input_eps_4;
float *__host_feature_input_eps_4;
float *__device_feature_input_eps_4;

float __device__ *feature_rst_4;
float *__host_feature_rst_4;
float *__device_feature_rst_4;

float __device__ *weight_4;
float *__host_weight_4;
float *__device_weight_4;

float __device__ *feature_input_out_5;
float *__host_feature_input_out_5;
float *__device_feature_input_out_5;

float __device__ *feature_input_eps_5;
float *__host_feature_input_eps_5;
float *__device_feature_input_eps_5;

float __device__ *feature_rst_5;
float *__host_feature_rst_5;
float *__device_feature_rst_5;

float __device__ *weight_5;
float *__host_weight_5;
float *__device_weight_5;

void __device__ init_feature(int32_t v, int32_t f);
void __device__ init_feature(int32_t v, int32_t f) {
	int32_t f_size = input_f_size;
	feature_input[f] = ((v+1)%10 - f%f_size%10)/10000.0;
}

//layer-1
void __device__ gin_1(int32_t src, int32_t dst, int32_t feat, float* A, float* C, int Feat_Size);
//template <typename EdgeWeightType>
void __device__ gpu_operator_body_0(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {

    gin_1(dst, src, feat, A, C, Feat_Size);
}
void __device__ gin_1(int32_t src, int32_t dst, int32_t feat, float* A, float* C, int Feat_Size) {
    gpu_runtime::writeAdd(&C[((dst * Feat_Size) + feat)], A[((src * Feat_Size) + feat)]);
}

void __device__ gin_1_nAtm(int32_t src, int32_t dst, int32_t feat, float* A, float* C, int Feat_Size);
//template <typename EdgeWeightType>
void __device__ gpu_operator_body_0_nAtm(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {
    
    gin_1_nAtm(dst, src, feat, A, C, Feat_Size);
}
void __device__ gin_1_nAtm(int32_t src, int32_t dst, int32_t feat, float* A, float* C, int Feat_Size) {
    C[((dst * Feat_Size) + feat)] += A[((src * Feat_Size) + feat)];
}

void __device__ eps_update_1(int32_t v, int32_t f);
void __device__ eps_update_1(int32_t v, int32_t f) {
    feature_input_eps_1[f] = (1 + eps) * feature_input[f] + feature_input_out_1[f];
}

//layer-2
void __device__ gin_2(int32_t src, int32_t dst, int32_t feat, float* A, float* C, int Feat_Size);
//template <typename EdgeWeightType>
void __device__ gpu_operator_body_1(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {

    gin_2(dst, src, feat, A, C, Feat_Size);
}
void __device__ gin_2(int32_t src, int32_t dst, int32_t feat, float* A, float* C, int Feat_Size) {
    gpu_runtime::writeAdd(&C[((dst * Feat_Size) + feat)], A[((src * Feat_Size) + feat)]);
}

void __device__ gin_2_nAtm(int32_t src, int32_t dst, int32_t feat, float* A, float* C, int Feat_Size);
//template <typename EdgeWeightType>
void __device__ gpu_operator_body_1_nAtm(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {
    
    gin_2_nAtm(dst, src, feat, A, C, Feat_Size);
}
void __device__ gin_2_nAtm(int32_t src, int32_t dst, int32_t feat, float* A, float* C, int Feat_Size) {
    C[((dst * Feat_Size) + feat)] += A[((src * Feat_Size) + feat)];
}

void __device__ eps_update_2(int32_t v, int32_t f);
void __device__ eps_update_2(int32_t v, int32_t f) {
    feature_input_eps_2[f] = (1 + eps) * feature_rst_1[f] + feature_input_out_2[f];
}

//layer-3
void __device__ gin_3(int32_t src, int32_t dst, int32_t feat, float* A, float* C, int Feat_Size);
//template <typename EdgeWeightType>
void __device__ gpu_operator_body_2(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {

    gin_3(dst, src, feat, A, C, Feat_Size);
}
void __device__ gin_3(int32_t src, int32_t dst, int32_t feat, float* A, float* C, int Feat_Size) {
    gpu_runtime::writeAdd(&C[((dst * Feat_Size) + feat)], A[((src * Feat_Size) + feat)]);
}

void __device__ gin_3_nAtm(int32_t src, int32_t dst, int32_t feat, float* A, float* C, int Feat_Size);
//template <typename EdgeWeightType>
void __device__ gpu_operator_body_2_nAtm(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {
    
    gin_3_nAtm(dst, src, feat, A, C, Feat_Size);
}
void __device__ gin_3_nAtm(int32_t src, int32_t dst, int32_t feat, float* A, float* C, int Feat_Size) {
    C[((dst * Feat_Size) + feat)] += A[((src * Feat_Size) + feat)];
}

void __device__ eps_update_3(int32_t v, int32_t f);
void __device__ eps_update_3(int32_t v, int32_t f) {
    feature_input_eps_3[f] = (1 + eps) * feature_rst_2[f] + feature_input_out_3[f];
}

//layer-4
void __device__ gin_4(int32_t src, int32_t dst, int32_t feat, float* A, float* C, int Feat_Size);
//template <typename EdgeWeightType>
void __device__ gpu_operator_body_3(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {

    gin_4(dst, src, feat, A, C, Feat_Size);
}
void __device__ gin_4(int32_t src, int32_t dst, int32_t feat, float* A, float* C, int Feat_Size) {
    gpu_runtime::writeAdd(&C[((dst * Feat_Size) + feat)], A[((src * Feat_Size) + feat)]);
}

void __device__ gin_4_nAtm(int32_t src, int32_t dst, int32_t feat, float* A, float* C, int Feat_Size);
//template <typename EdgeWeightType>
void __device__ gpu_operator_body_3_nAtm(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {
    
    gin_4_nAtm(dst, src, feat, A, C, Feat_Size);
}
void __device__ gin_4_nAtm(int32_t src, int32_t dst, int32_t feat, float* A, float* C, int Feat_Size) {
    C[((dst * Feat_Size) + feat)] += A[((src * Feat_Size) + feat)];
}

void __device__ eps_update_4(int32_t v, int32_t f);
void __device__ eps_update_4(int32_t v, int32_t f) {
    feature_input_eps_4[f] = (1 + eps) * feature_rst_3[f] + feature_input_out_4[f];
}

// layer-5
void __device__ gin_5(int32_t src, int32_t dst, int32_t feat, float* A, float* C, int Feat_Size);
//template <typename EdgeWeightType>
void __device__ gpu_operator_body_4(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {

    gin_5(dst, src, feat, A, C, Feat_Size);
}
void __device__ gin_5(int32_t src, int32_t dst, int32_t feat, float* A, float* C, int Feat_Size) {
    gpu_runtime::writeAdd(&C[((dst * Feat_Size) + feat)], A[((src * Feat_Size) + feat)]);
}

void __device__ gin_5_nAtm(int32_t src, int32_t dst, int32_t feat, float* A, float* C, int Feat_Size);
//template <typename EdgeWeightType>
void __device__ gpu_operator_body_4_nAtm(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {
    
    gin_5_nAtm(dst, src, feat, A, C, Feat_Size);
}
void __device__ gin_5_nAtm(int32_t src, int32_t dst, int32_t feat, float* A, float* C, int Feat_Size) {
    C[((dst * Feat_Size) + feat)] += A[((src * Feat_Size) + feat)];
}

void __device__ eps_update_5(int32_t v, int32_t f);
void __device__ eps_update_5(int32_t v, int32_t f) {
    feature_input_eps_5[f] = (1 + eps) * feature_rst_4[f] + feature_input_out_5[f];
}

int __host__ main(int argc, char* argv[]) {
    cudaError_t cuda_error;
    cublasStatus_t cublas_error;
    // Initialize CUDA
    cuda_error = cudaSetDevice(0);
    if (cuda_error != cudaSuccess) {
        cout << "Error: cudaSetDevice failed!" << endl;
        return EXIT_FAILURE;
    }

    // Initialize CUBLAS
    // cublasHandle_t cublas_handle;

    const int GPUNUM = 1;
    cublasHandle_t* cublasHs = new cublasHandle_t[GPUNUM];
    for (int i = 0; i < GPUNUM; i++) {
        cublas_error = cublasCreate(&cublasHs[i]);
        if (cublas_error != CUBLAS_STATUS_SUCCESS) {
            cout << "Error: cublasCreate failed!" << endl;
            return EXIT_FAILURE;
        }
    }

	__delta_param = 1;
	gpu_runtime::load_graph(__host_edges, argv[1], false);
	cudaMemcpyToSymbol(edges, &__host_edges, sizeof(__host_edges), 0, cudaMemcpyHostToDevice);

    __host_edges__transposed = gpu_runtime::builtin_transpose(__host_edges);
	CUDA_CALL(cudaMemcpyToSymbol(edges__transposed, &__host_edges__transposed, sizeof(__host_edges__transposed), 0, cudaMemcpyHostToDevice));

	__host_M = 128;
	cudaMemcpyToSymbol(M, &__host_M, sizeof(int32_t), 0, cudaMemcpyHostToDevice);
	__host_input_f_size = std::stoi(argv[2]);
	cudaMemcpyToSymbol(input_f_size, &__host_input_f_size, sizeof(int32_t), 0, cudaMemcpyHostToDevice);
	__host_hidden_f_size = std::stoi(argv[3]);
	cudaMemcpyToSymbol(hidden_f_size, &__host_hidden_f_size, sizeof(int32_t), 0, cudaMemcpyHostToDevice);
	__host_output_f_size = std::stoi(argv[4]);
	cudaMemcpyToSymbol(output_f_size, &__host_output_f_size, sizeof(int32_t), 0, cudaMemcpyHostToDevice);
	__host_vector_size = __host_edges__transposed.num_vertices;
    cudaMemcpyToSymbol(vector_size, &__host_vector_size, sizeof(int32_t), 0, cudaMemcpyHostToDevice);
	
    sz_fin = __host_input_f_size * __host_vector_size;
    sz_fhid = __host_hidden_f_size * __host_vector_size;
    sz_fout = __host_output_f_size * __host_vector_size;

    sz_w1 = __host_input_f_size * __host_hidden_f_size;
    sz_w2 = __host_hidden_f_size * __host_hidden_f_size;
    sz_w3 = __host_hidden_f_size * __host_output_f_size;
    
    //layer-1
	cudaMalloc(&__device_feature_input, sz_fin * sizeof(float));
    cudaMemset(__device_feature_input, 0, sz_fin * sizeof(float));
	cudaMemcpyToSymbol(feature_input, &__device_feature_input, sizeof(float*), 0);
	__host_feature_input = new float[sz_fin];

	cudaMalloc(&__device_feature_input_out_1, sz_fin * sizeof(float));
    cudaMemset(__device_feature_input_out_1, 0, sz_fin * sizeof(float));
	cudaMemcpyToSymbol(feature_input_out_1, &__device_feature_input_out_1, sizeof(float*), 0);
	__host_feature_input_out_1 = new float[sz_fin];

	cudaMalloc(&__device_feature_input_eps_1, sz_fin * sizeof(float));
    cudaMemset(__device_feature_input_eps_1, 0, sz_fin * sizeof(float));
	cudaMemcpyToSymbol(feature_input_eps_1, &__device_feature_input_eps_1, sizeof(float*), 0);
	__host_feature_input_eps_1 = new float[sz_fin];

	cudaMalloc(&__device_feature_rst_1, sz_fhid * sizeof(float));
    cudaMemset(__device_feature_rst_1, 0, sz_fhid * sizeof(float));
	cudaMemcpyToSymbol(feature_rst_1, &__device_feature_rst_1, sizeof(float*), 0);
	__host_feature_rst_1 = new float[sz_fhid];

	cudaMalloc(&__device_weight_1, sz_w1 * sizeof(float));
    cudaMemset(__device_weight_1, 0, sz_w1 * sizeof(float));
	cudaMemcpyToSymbol(weight_1, &__device_weight_1, sizeof(float*), 0);
	__host_weight_1 = new float[sz_w1];

    //layer-2
	cudaMalloc(&__device_feature_input_out_2, sz_fhid * sizeof(float));
    cudaMemset(__device_feature_input_out_2, 0, sz_fhid * sizeof(float));
	cudaMemcpyToSymbol(feature_input_out_2, &__device_feature_input_out_2, sizeof(float*), 0);
	__host_feature_input_out_2 = new float[sz_fhid];

	cudaMalloc(&__device_feature_input_eps_2, sz_fhid * sizeof(float));
    cudaMemset(__device_feature_input_eps_2, 0, sz_fhid * sizeof(float));
	cudaMemcpyToSymbol(feature_input_eps_2, &__device_feature_input_eps_2, sizeof(float*), 0);
	__host_feature_input_eps_2 = new float[sz_fhid];

	cudaMalloc(&__device_feature_rst_2, sz_fhid * sizeof(float));
    cudaMemset(__device_feature_rst_2, 0, sz_fhid * sizeof(float));
	cudaMemcpyToSymbol(feature_rst_2, &__device_feature_rst_2, sizeof(float*), 0);
	__host_feature_rst_2 = new float[sz_fhid];

	cudaMalloc(&__device_weight_2, sz_w2 * sizeof(float));
    cudaMemset(__device_weight_2, 0, sz_w2 * sizeof(float));
	cudaMemcpyToSymbol(weight_2, &__device_weight_2, sizeof(float*), 0);
	__host_weight_2 = new float[sz_w2];

    //layer-3
	cudaMalloc(&__device_feature_input_out_3, sz_fhid * sizeof(float));
    cudaMemset(__device_feature_input_out_3, 0, sz_fhid * sizeof(float));
	cudaMemcpyToSymbol(feature_input_out_3, &__device_feature_input_out_3, sizeof(float*), 0);
	__host_feature_input_out_3 = new float[sz_fhid];

	cudaMalloc(&__device_feature_input_eps_3, sz_fhid * sizeof(float));
    cudaMemset(__device_feature_input_eps_3, 0, sz_fhid * sizeof(float));
	cudaMemcpyToSymbol(feature_input_eps_3, &__device_feature_input_eps_3, sizeof(float*), 0);
	__host_feature_input_eps_3 = new float[sz_fhid];

	cudaMalloc(&__device_feature_rst_3, sz_fhid * sizeof(float));
    cudaMemset(__device_feature_rst_3, 0, sz_fhid * sizeof(float));
	cudaMemcpyToSymbol(feature_rst_3, &__device_feature_rst_3, sizeof(float*), 0);
	__host_feature_rst_3 = new float[sz_fhid];

	cudaMalloc(&__device_weight_3, sz_w2 * sizeof(float));
    cudaMemset(__device_weight_3, 0, sz_w2 * sizeof(float));
	cudaMemcpyToSymbol(weight_3, &__device_weight_3, sizeof(float*), 0);
	__host_weight_3 = new float[sz_w2];

    //layer-4
	cudaMalloc(&__device_feature_input_out_4, sz_fhid * sizeof(float));
    cudaMemset(__device_feature_input_out_4, 0, sz_fhid * sizeof(float));
	cudaMemcpyToSymbol(feature_input_out_4, &__device_feature_input_out_4, sizeof(float*), 0);
	__host_feature_input_out_4 = new float[sz_fhid];

	cudaMalloc(&__device_feature_input_eps_4, sz_fhid * sizeof(float));
    cudaMemset(__device_feature_input_eps_4, 0, sz_fhid * sizeof(float));
	cudaMemcpyToSymbol(feature_input_eps_4, &__device_feature_input_eps_4, sizeof(float*), 0);
	__host_feature_input_eps_4 = new float[sz_fhid];

	cudaMalloc(&__device_feature_rst_4, sz_fhid * sizeof(float));
    cudaMemset(__device_feature_rst_4, 0, sz_fhid * sizeof(float));
	cudaMemcpyToSymbol(feature_rst_4, &__device_feature_rst_4, sizeof(float*), 0);
	__host_feature_rst_4 = new float[sz_fhid];

	cudaMalloc(&__device_weight_4, sz_w2 * sizeof(float));
    cudaMemset(__device_weight_4, 0, sz_w2 * sizeof(float));
	cudaMemcpyToSymbol(weight_4, &__device_weight_4, sizeof(float*), 0);
	__host_weight_4 = new float[sz_w2];

    //layer-5
	CUDA_CALL(cudaMalloc(&__device_feature_input_out_5, sz_fhid * sizeof(float)));
    CUDA_CALL(cudaMemset(__device_feature_input_out_5, 0, sz_fhid * sizeof(float)));
	CUDA_CALL(cudaMemcpyToSymbol(feature_input_out_5, &__device_feature_input_out_5, sizeof(float*), 0));
	__host_feature_input_out_5 = new float[sz_fhid];

	CUDA_CALL(cudaMalloc(&__device_feature_input_eps_5, sz_fhid * sizeof(float)));
    CUDA_CALL(cudaMemset(__device_feature_input_eps_5, 0, sz_fhid * sizeof(float)));
	CUDA_CALL(cudaMemcpyToSymbol(feature_input_eps_5, &__device_feature_input_eps_5, sizeof(float*), 0));
	__host_feature_input_eps_5 = new float[sz_fhid];

	CUDA_CALL(cudaMalloc(&__device_feature_rst_5, sz_fout * sizeof(float)));
    CUDA_CALL(cudaMemset(__device_feature_rst_5, 0, sz_fout * sizeof(float)));
	CUDA_CALL(cudaMemcpyToSymbol(feature_rst_5, &__device_feature_rst_5, sizeof(float*), 0));
	__host_feature_rst_5 = new float[sz_fout];

	CUDA_CALL(cudaMalloc(&__device_weight_5, sz_w3 * sizeof(float)));
    CUDA_CALL(cudaMemset(__device_weight_5, 0, sz_w3 * sizeof(float)));
	CUDA_CALL(cudaMemcpyToSymbol(weight_5, &__device_weight_5, sizeof(float*), 0));
	__host_weight_5 = new float[sz_w3];

    // printf("init __host_weight_1:%d %d %d\n", __host_input_f_size, __host_hidden_f_size, __host_hidden_f_size*__host_input_f_size);
    for (int32_t i = 0; i < __host_input_f_size; i++) {
        for (int32_t j = 0; j < __host_hidden_f_size; j++) {
            __host_weight_1[((i * __host_hidden_f_size) + j)] = (j%10 - i%10)/10000.0;
            CUDA_CALL(cudaMemcpy(__device_weight_1 + ((i * __host_hidden_f_size) + j), __host_weight_1 + ((i * __host_hidden_f_size) + j), sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    // printf("init __host_weight_2:%d %d %d\n", __host_hidden_f_size, __host_hidden_f_size, __host_hidden_f_size*__host_hidden_f_size);
    for (int32_t i = 0; i < __host_hidden_f_size; i++) {
        for (int32_t j = 0; j < __host_hidden_f_size; j++) {
            __host_weight_2[((i * __host_hidden_f_size) + j)] = (i%10 - j%10)/10000.0;
            CUDA_CALL(cudaMemcpy(__device_weight_2 + ((i * __host_hidden_f_size) + j), __host_weight_2 + ((i * __host_hidden_f_size) + j), sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    // printf("init __host_weight_3:%d %d %d\n", __host_hidden_f_size, __host_hidden_f_size, __host_hidden_f_size*__host_hidden_f_size);
    for (int32_t i = 0; i < __host_hidden_f_size; i++) {
        for (int32_t j = 0; j < __host_hidden_f_size; j++) {
            __host_weight_3[((i * __host_hidden_f_size) + j)] = (j%10 - i%10)/10000.0;
            CUDA_CALL(cudaMemcpy(__device_weight_3 + ((i * __host_hidden_f_size) + j), __host_weight_3 + ((i * __host_hidden_f_size) + j), sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    // printf("init __host_weight_4:%d %d %d\n", __host_hidden_f_size, __host_hidden_f_size, __host_hidden_f_size*__host_hidden_f_size);
    for (int32_t i = 0; i < __host_hidden_f_size; i++) {
        for (int32_t j = 0; j < __host_hidden_f_size; j++) {
            __host_weight_4[((i * __host_hidden_f_size) + j)] = (i%10 - j%10)/10000.0;
            CUDA_CALL(cudaMemcpy(__device_weight_4 + ((i * __host_hidden_f_size) + j), __host_weight_4 + ((i * __host_hidden_f_size) + j), sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    // printf("init __host_weight_5:%d %d %d\n", __host_hidden_f_size, __host_output_f_size, __host_hidden_f_size*__host_output_f_size);
    for (int32_t i = 0; i < __host_hidden_f_size; i++) {
        for (int32_t j = 0; j < __host_output_f_size; j++) {
            __host_weight_5[((i * __host_output_f_size) + j)] = (i%10 - j%10)/10000.0;
            CUDA_CALL(cudaMemcpy(__device_weight_5 + ((i * __host_output_f_size) + j), __host_weight_5 + ((i * __host_output_f_size) + j), sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    cuda_error = cudaGetLastError();

    if (cuda_error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(cuda_error));
    }

    // decide load balance schedule
	gpu_runtime::gnn_load_balance_type lb_c[5];
    int32_t ef_num_cta[5]={0,0,0,0,0};
    int32_t ef_feat_size[5];
    ef_feat_size[0] = __host_input_f_size;
    ef_feat_size[1] = __host_hidden_f_size;
    ef_feat_size[2] = __host_hidden_f_size;
    ef_feat_size[3] = __host_hidden_f_size;
    ef_feat_size[4] = __host_hidden_f_size;
    int32_t group_size[5]={std::stoi(argv[9]),std::stoi(argv[13]),std::stoi(argv[17]),std::stoi(argv[21]),std::stoi(argv[25])};
    int32_t par_tiling[5]={std::stoi(argv[10]),std::stoi(argv[14]),std::stoi(argv[18]),std::stoi(argv[22]),std::stoi(argv[26])};
    gpu_runtime::uGrapher_init<gpu_operator_body_0, gpu_operator_body_0_nAtm>(
        0, argv[8], lb_c, __host_edges__transposed, ef_num_cta, 
        ef_feat_size, group_size, par_tiling);
    gpu_runtime::uGrapher_init<gpu_operator_body_1, gpu_operator_body_1_nAtm>(
        1, argv[12], lb_c, __host_edges__transposed, ef_num_cta, 
        ef_feat_size, group_size, par_tiling);
    gpu_runtime::uGrapher_init<gpu_operator_body_2, gpu_operator_body_2_nAtm>(
        2, argv[16], lb_c, __host_edges__transposed, ef_num_cta, 
        ef_feat_size, group_size, par_tiling);
    gpu_runtime::uGrapher_init<gpu_operator_body_3, gpu_operator_body_3_nAtm>(
        3, argv[20], lb_c, __host_edges__transposed, ef_num_cta, 
        ef_feat_size, group_size, par_tiling);
    gpu_runtime::uGrapher_init<gpu_operator_body_4, gpu_operator_body_4_nAtm>(
        4, argv[24], lb_c, __host_edges__transposed, ef_num_cta, 
        ef_feat_size, group_size, par_tiling);

    gpu_runtime::f_vertex_set_apply_host<init_feature>(__host_edges__transposed, __host_input_f_size);
    
    int32_t vf_num_cta[5];
	int32_t vf_feat_size[5];
    vf_feat_size[0] = __host_input_f_size;
    vf_feat_size[1] = __host_hidden_f_size;
    vf_feat_size[2] = __host_hidden_f_size;
    vf_feat_size[3] = __host_hidden_f_size;
    vf_feat_size[4] = __host_hidden_f_size;
    gpu_runtime::f_vertex_set_apply_info(__host_edges__transposed, vf_num_cta[0], vf_feat_size[0]);
    gpu_runtime::f_vertex_set_apply_info(__host_edges__transposed, vf_num_cta[1], vf_feat_size[1]);
    gpu_runtime::f_vertex_set_apply_info(__host_edges__transposed, vf_num_cta[2], vf_feat_size[2]);
    gpu_runtime::f_vertex_set_apply_info(__host_edges__transposed, vf_num_cta[3], vf_feat_size[3]);
    gpu_runtime::f_vertex_set_apply_info(__host_edges__transposed, vf_num_cta[4], vf_feat_size[4]);
    
    float time = 0;
    int cnt = 20;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

	for (int32_t trail = 0; trail < cnt; trail++) {
        //layer-1
        {
            gpu_runtime::uGrapher_exec(0, lb_c, ef_num_cta, 
                __host_edges__transposed, __device_feature_input, 0, __device_feature_input_out_1, 
                ef_feat_size, group_size, par_tiling);
            // cudaDeviceSynchronize();
        }

        {
            gpu_runtime::f_vertex_set_apply_kernel<eps_update_1><<<vf_num_cta[0], CTA_SIZE>>>(__host_edges__transposed, vf_feat_size[0]);
            // cudaDeviceSynchronize();
        }

        gpu_runtime::matmul_NN(cublasHs, __device_feature_input_eps_1, __device_weight_1, __device_feature_rst_1, __host_vector_size, __host_hidden_f_size, __host_input_f_size);

        //layer-2
        {
            gpu_runtime::uGrapher_exec(1, lb_c, ef_num_cta, 
                __host_edges__transposed, __device_feature_rst_1, 0, __device_feature_input_out_2, 
                ef_feat_size, group_size, par_tiling);
            // cudaDeviceSynchronize();
        }

        {
            gpu_runtime::f_vertex_set_apply_kernel<eps_update_2><<<vf_num_cta[1], CTA_SIZE>>>(__host_edges__transposed, vf_feat_size[1]);
            // cudaDeviceSynchronize();
        }

        gpu_runtime::matmul_NN(cublasHs, __device_feature_input_eps_2, __device_weight_2, __device_feature_rst_2, __host_vector_size, __host_hidden_f_size, __host_hidden_f_size);

        //layer-3
        {
            gpu_runtime::uGrapher_exec(2, lb_c, ef_num_cta, 
                __host_edges__transposed, __device_feature_rst_2, 0, __device_feature_input_out_3, 
                ef_feat_size, group_size, par_tiling);
            // cudaDeviceSynchronize();
        }

        {
            gpu_runtime::f_vertex_set_apply_kernel<eps_update_3><<<vf_num_cta[2], CTA_SIZE>>>(__host_edges__transposed, vf_feat_size[2]);
            // cudaDeviceSynchronize();
        }

        gpu_runtime::matmul_NN(cublasHs, __device_feature_input_eps_3, __device_weight_3, __device_feature_rst_3, __host_vector_size, __host_hidden_f_size, __host_hidden_f_size);

        //layer-4
        {
            gpu_runtime::uGrapher_exec(3, lb_c, ef_num_cta, 
                __host_edges__transposed, __device_feature_rst_3, 0, __device_feature_input_out_4, 
                ef_feat_size, group_size, par_tiling);
            // cudaDeviceSynchronize();
        }

        {
            gpu_runtime::f_vertex_set_apply_kernel<eps_update_4><<<vf_num_cta[3], CTA_SIZE>>>(__host_edges__transposed, vf_feat_size[3]);
            // cudaDeviceSynchronize();
        }

        gpu_runtime::matmul_NN(cublasHs, __device_feature_input_eps_4, __device_weight_4, __device_feature_rst_4, __host_vector_size, __host_hidden_f_size, __host_hidden_f_size);

        //layer-5
        {
            gpu_runtime::uGrapher_exec(4, lb_c, ef_num_cta, 
                __host_edges__transposed, __device_feature_rst_4, 0, __device_feature_input_out_5, 
                ef_feat_size, group_size, par_tiling);
            // cudaDeviceSynchronize();
        }

        {
            gpu_runtime::f_vertex_set_apply_kernel<eps_update_5><<<vf_num_cta[4], CTA_SIZE>>>(__host_edges__transposed, vf_feat_size[4]);
            // cudaDeviceSynchronize();
        }

        gpu_runtime::matmul_NN(cublasHs, __device_feature_input_eps_5, __device_weight_5, __device_feature_rst_5, __host_vector_size, __host_output_f_size, __host_hidden_f_size);
        // gpu_runtime::matmul_NN(cublasHs, __device_feature_rst_4, __device_weight_5, __device_feature_rst_5, __host_vector_size, __host_output_f_size, __host_hidden_f_size);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    time = time / cnt;
    cout << "time: " << time << " msec" << endl;

    cuda_error = cudaGetLastError();

    if (cuda_error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(cuda_error));
    }

    std::fstream fp;
	fp.open(argv[27], std::ios::out|std::ios::app);
	fp << time << std::endl;
	fp.close();

    cudaFree(__device_feature_input);
    cudaFree(__device_feature_input_out_1);
    cudaFree(__device_feature_input_out_2);
    cudaFree(__device_feature_input_out_3);
    cudaFree(__device_feature_input_out_4);
    cudaFree(__device_feature_input_out_5);
    cudaFree(__device_feature_input_eps_1);
    cudaFree(__device_feature_input_eps_2);
    cudaFree(__device_feature_input_eps_3);
    cudaFree(__device_feature_input_eps_4);
    cudaFree(__device_feature_input_eps_5);
    cudaFree(__device_weight_1);
    cudaFree(__device_weight_2);
    cudaFree(__device_weight_3);
    cudaFree(__device_weight_4);
    cudaFree(__device_weight_5);
    cudaFree(__device_feature_rst_1);
    cudaFree(__device_feature_rst_2);
    cudaFree(__device_feature_rst_3);
    cudaFree(__device_feature_rst_4);
    cudaFree(__device_feature_rst_5);

    delete __host_feature_input;
    delete __host_feature_input_out_1;
    delete __host_feature_input_out_2;
    delete __host_feature_input_out_3;
    delete __host_feature_input_out_4;
    delete __host_feature_input_out_5;
    delete __host_feature_input_eps_1;
    delete __host_feature_input_eps_2;
    delete __host_feature_input_eps_3;
    delete __host_feature_input_eps_4;
    delete __host_feature_input_eps_5;
    delete __host_weight_1;
    delete __host_weight_2;
    delete __host_weight_3;
    delete __host_weight_4;
    delete __host_weight_5;
    delete __host_feature_rst_1;
    delete __host_feature_rst_2;
    delete __host_feature_rst_3;
    delete __host_feature_rst_4;
    delete __host_feature_rst_5;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    for (int i = 0; i < GPUNUM; i++) {
		cublasDestroy(cublasHs[i]);
	}

	delete cublasHs;

    return 0;
}