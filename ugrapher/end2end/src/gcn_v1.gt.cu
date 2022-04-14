#include "gpu_intrinsics.h"
#include <cooperative_groups.h>
using namespace cooperative_groups;

// total feature size
int32_t sz_fin;		    // (model_fin  * model_vertex)
int32_t sz_fhid;		// (model_fhid * model_vertex)
int32_t sz_fout;		// (model_fout * model_vertex)

// weight size
int32_t sz_w1;          // (model_fin * model_fhid)
int32_t sz_w2;          // (model_fhid * model_fout)

// norm weight
// #define sz_nw       (size_t(model_vertex) * size_t(model_vertex))   
int32_t sz_edge_nw;     // (size_t(model_edge))

int32_t __delta_param;
gpu_runtime::GraphT __device__ edges;
gpu_runtime::GraphT __host_edges;
gpu_runtime::GraphT __device__ edges__transposed;
gpu_runtime::GraphT __host_edges__transposed;
// gpu_runtime::GraphT<char> __device__ edges;
// gpu_runtime::GraphT<char> __host_edges;
// gpu_runtime::GraphT<char> __device__ edges__transposed;
// gpu_runtime::GraphT<char> __host_edges__transposed;

int32_t __device__ input_f_size; 
int32_t __host_input_f_size;

int32_t __device__ hidden_f_size; 
int32_t __host_hidden_f_size;

int32_t __device__ output_f_size; 
int32_t __host_output_f_size;

int32_t __device__ vector_size; 
int32_t __host_vector_size;

int32_t __device__ edge_size;
int32_t __host_edge_size;

float __device__ *feature_input;
float *__host_feature_input;
float *__device_feature_input;

float __device__ *feature_input_fc_1;
float *__host_feature_input_fc_1;
float *__device_feature_input_fc_1;

float __device__ *feature_input_fc_2;
float *__host_feature_input_fc_2;
float *__device_feature_input_fc_2;

float __device__ *feature_input_out_1;
float *__host_feature_input_out_1;
float *__device_feature_input_out_1;

float __device__ *feature_input_out_2;
float *__host_feature_input_out_2;
float *__device_feature_input_out_2;

float __device__ *feature_rst_1;
float *__host_feature_rst_1;
float *__device_feature_rst_1;

float __device__ *feature_rst_2;
float *__host_feature_rst_2;
float *__device_feature_rst_2;

float __device__ *weight_fc_1;
float *__host_weight_fc_1;
float *__device_weight_fc_1;

float __device__ *weight_fc_2;
float *__host_weight_fc_2;
float *__device_weight_fc_2;

float __device__ *weight_norm;
float *__host_weight_norm;
float *__device_weight_norm;

void __device__ sum_1(int32_t src, int32_t dst, int32_t edge, int32_t feat, float* A, float* B, float* C, int Feat_Size);
void __device__ sum_1_nAtm(int32_t src, int32_t dst, int32_t edge, int32_t feat, float* A, float* B, float* C, int Feat_Size);
void __device__ sum_2(int32_t src, int32_t dst, int32_t edge, int32_t feat, float* A, float* B, float* C, int Feat_Size);
void __device__ sum_2_nAtm(int32_t src, int32_t dst, int32_t edge, int32_t feat, float* A, float* B, float* C, int Feat_Size);

void __device__ get_rst_1(int32_t v, int32_t f);
void __device__ get_rst_2(int32_t v, int32_t f);
void __device__ init_feature(int32_t v, int32_t f);

//template <typename EdgeWeightType>
void __device__ gpu_operator_body_0(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {
	// Body of the actual operator
	sum_1(dst, src, edge, feat, A, B, C, Feat_Size);
}

//template <typename EdgeWeightType>
void __device__ gpu_operator_body_0_nAtm(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {
	// Body of the actual operator
	sum_1_nAtm(dst, src, edge, feat, A, B, C, Feat_Size);
}

//template <typename EdgeWeightType>
void __device__ gpu_operator_body_1(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {
	// Body of the actual operator
	sum_2(dst, src, edge, feat, A, B, C, Feat_Size);
}

//template <typename EdgeWeightType>
void __device__ gpu_operator_body_1_nAtm(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {
	// Body of the actual operator
	sum_2_nAtm(dst, src, edge, feat, A, B, C, Feat_Size);
}

void __device__ sum_1(int32_t src, int32_t dst, int32_t edge, int32_t feat, float* A, float* B, float* C, int Feat_Size) {
    gpu_runtime::writeAdd(&C[((dst * Feat_Size) + feat)], A[((src * Feat_Size) + feat)] * B[edge]);
}

void __device__ sum_1_nAtm(int32_t src, int32_t dst, int32_t edge, int32_t feat, float* A, float* B, float* C, int Feat_Size) {
    C[((dst * Feat_Size) + feat)] += A[((src * Feat_Size) + feat)] * B[edge];
}

void __device__ sum_2(int32_t src, int32_t dst, int32_t edge, int32_t feat, float* A, float* B, float* C, int Feat_Size) {
    gpu_runtime::writeAdd(&C[((dst * Feat_Size) + feat)], A[((src * Feat_Size) + feat)] * B[edge]);
}

void __device__ sum_2_nAtm(int32_t src, int32_t dst, int32_t edge, int32_t feat, float* A, float* B, float* C, int Feat_Size) {
    C[((dst * Feat_Size) + feat)] += A[((src * Feat_Size) + feat)] * B[edge];
}

void __device__ init_feature(int32_t v, int32_t f) {
	int32_t f_size = input_f_size;
	feature_input[f] = (v + f % f_size + 1) / 10000.0;
}

void __device__ get_rst_1(int32_t v, int32_t f) {
    feature_rst_1[f] = (feature_input_out_1[f] > 0) ? feature_input_out_1[f] : 0;
}

void __device__ get_rst_2(int32_t v, int32_t f) {
    feature_rst_2[f] = (feature_input_out_2[f] > 0) ? feature_input_out_2[f] : 0;
}

__global__ void initWeightNorm_ker(float *data, size_t numElements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numElements) {
        size_t src_id, dst_id;
        src_id = edges__transposed.d_edge_src[tid];
        dst_id = edges__transposed.d_edge_dst[tid];

        size_t src_deg, dst_deg;
        src_deg = edges__transposed.d_get_degree(src_id);
        dst_deg = edges__transposed.d_get_degree(dst_id);
        data[tid] = pow(src_deg, -0.5) * pow(dst_deg, -0.5);
    }
 }

// used to init norm weight in GCN
void __host__ initWeightNorm(float * weight_norm) {
    dim3 gridDim;
    dim3 blockDim;
 
    blockDim.x = 1024;
    size_t numElements = sz_edge_nw;

    gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;
 
    initWeightNorm_ker <<< gridDim, blockDim >>> (weight_norm, numElements);
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

    // printf("Graph loaded\n");
    // printf("Graph: %d nodes, %d edges\n", __host_edges.num_vertices, __host_edges.num_edges);

	__host_edges__transposed = gpu_runtime::builtin_transpose(__host_edges);
	CUDA_CALL(cudaMemcpyToSymbol(edges__transposed, &__host_edges__transposed, sizeof(__host_edges__transposed), 0, cudaMemcpyHostToDevice));

    cuda_error = cudaGetLastError();

    if (cuda_error != cudaSuccess) {
        printf("172 CUDA Error: %s\n", cudaGetErrorString(cuda_error));
    }

    __host_input_f_size = std::stoi(argv[2]);
	cudaMemcpyToSymbol(input_f_size, &__host_input_f_size, sizeof(int32_t), 0, cudaMemcpyHostToDevice);
	__host_hidden_f_size = std::stoi(argv[3]);
	cudaMemcpyToSymbol(hidden_f_size, &__host_hidden_f_size, sizeof(int32_t), 0, cudaMemcpyHostToDevice);
	__host_output_f_size = std::stoi(argv[4]);
	cudaMemcpyToSymbol(output_f_size, &__host_output_f_size, sizeof(int32_t), 0, cudaMemcpyHostToDevice);
	__host_vector_size = __host_edges__transposed.num_vertices;
	cudaMemcpyToSymbol(vector_size, &__host_vector_size, sizeof(int32_t), 0, cudaMemcpyHostToDevice);
    __host_edge_size = __host_edges__transposed.num_edges;
	cudaMemcpyToSymbol(edge_size, &__host_edge_size, sizeof(int32_t), 0, cudaMemcpyHostToDevice);
	
    sz_fin = __host_input_f_size * __host_vector_size;
    sz_fhid = __host_hidden_f_size * __host_vector_size;
    sz_fout = __host_output_f_size * __host_vector_size;

    sz_w1 = __host_input_f_size * __host_hidden_f_size;
    sz_w2 = __host_hidden_f_size * __host_output_f_size;

    sz_edge_nw = __host_edge_size;

    //layer-1
    cudaMalloc(&__device_feature_input, sz_fin * sizeof(float));
    cudaMemset(__device_feature_input, 0, sz_fin * sizeof(float));
    cudaMemcpyToSymbol(feature_input, &__device_feature_input, sizeof(float*), 0);
	__host_feature_input = new float[sz_fin];
	
    cudaMalloc(&__device_feature_input_fc_1, sz_fhid * sizeof(float));
    cudaMemset(__device_feature_input_fc_1, 0, sz_fhid * sizeof(float));
    cudaMemcpyToSymbol(feature_input_fc_1, &__device_feature_input_fc_1, sizeof(float*), 0);
    __host_feature_input_fc_1 = new float[sz_fhid];

    cudaMalloc(&__device_feature_input_out_1, sz_fhid * sizeof(float));
    cudaMemset(__device_feature_input_out_1, 0, sz_fhid * sizeof(float));
	cudaMemcpyToSymbol(feature_input_out_1, &__device_feature_input_out_1, sizeof(float*), 0);
	__host_feature_input_out_1 = new float[sz_fhid];

    cudaMalloc(&__device_feature_rst_1, sz_fhid * sizeof(float));
    cudaMemset(__device_feature_rst_1, 0, sz_fhid * sizeof(float));
    cudaMemcpyToSymbol(feature_rst_1, &__device_feature_rst_1, sizeof(float*), 0);
    __host_feature_rst_1 = new float[sz_fhid];

    cudaMalloc(&__device_weight_fc_1, sz_w1 * sizeof(float));
    cudaMemset(__device_weight_fc_1, 0, sz_w1 * sizeof(float));
    cudaMemcpyToSymbol(weight_fc_1, &__device_weight_fc_1, sizeof(float*), 0);
    __host_weight_fc_1 = new float[sz_w1];

    //layer-2
    cudaMalloc(&__device_feature_input_fc_2, sz_fout * sizeof(float));
    cudaMemset(__device_feature_input_fc_2, 0, sz_fout * sizeof(float));
    cudaMemcpyToSymbol(feature_input_fc_2, &__device_feature_input_fc_2, sizeof(float*), 0);
    __host_feature_input_fc_2 = new float[sz_fout];

    cudaMalloc(&__device_feature_input_out_2, sz_fout * sizeof(float));
    cudaMemset(__device_feature_input_out_2, 0, sz_fout * sizeof(float));
    cudaMemcpyToSymbol(feature_input_out_2, &__device_feature_input_out_2, sizeof(float*), 0);
    __host_feature_input_out_2 = new float[sz_fout];

    cudaMalloc(&__device_feature_rst_2, sz_fout * sizeof(float));
    cudaMemset(__device_feature_rst_2, 0, sz_fout * sizeof(float));
    cudaMemcpyToSymbol(feature_rst_2, &__device_feature_rst_2, sizeof(float*), 0);
    __host_feature_rst_2 = new float[sz_fout];

    cudaMalloc(&__device_weight_fc_2, sz_w2 * sizeof(float));
    cudaMemset(__device_weight_fc_2, 0, sz_w2 * sizeof(float));
    cudaMemcpyToSymbol(weight_fc_2, &__device_weight_fc_2, sizeof(float*), 0);
    __host_weight_fc_2 = new float[sz_w2];

    cudaMalloc(&__device_weight_norm, sz_edge_nw * sizeof(float));
    cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        printf("295 CUDA Error: %s\n", cudaGetErrorString(cuda_error));
        return -1;
    }
    cudaMemset(__device_weight_norm, 0, sz_edge_nw * sizeof(float));
    cudaMemcpyToSymbol(weight_norm, &__device_weight_norm, sizeof(float*), 0);
    __host_weight_norm = new float[sz_edge_nw];

    initGPUData(__device_weight_fc_1, sz_w1, 0.001);
    initGPUData(__device_weight_fc_2, sz_w2, 0.002);
    // initGPUData(__device_weight_norm, __host_vector_size * __host_vector_size, 0.3);
    initWeightNorm(__device_weight_norm);

    cuda_error = cudaGetLastError();

    if (cuda_error != cudaSuccess) {
        printf("321 CUDA Error: %s\n", cudaGetErrorString(cuda_error));
        return -1;
    }

    // decide load balance schedule
	gpu_runtime::gnn_load_balance_type lb_c[2];
    int32_t ef_num_cta[2]={0,0};
    int32_t ef_feat_size[2];
    ef_feat_size[0] = __host_hidden_f_size;
    ef_feat_size[1] = __host_output_f_size;
    int32_t group_size[2]={std::stoi(argv[9]),std::stoi(argv[13])};
    int32_t par_tiling[2]={std::stoi(argv[10]),std::stoi(argv[14])};
    gpu_runtime::uGrapher_init<gpu_operator_body_0, gpu_operator_body_0_nAtm>(
        0, argv[8], lb_c, __host_edges__transposed, ef_num_cta, 
        ef_feat_size, group_size, par_tiling);
    gpu_runtime::uGrapher_init<gpu_operator_body_1, gpu_operator_body_1_nAtm>(
        1, argv[12], lb_c, __host_edges__transposed, ef_num_cta, 
        ef_feat_size, group_size, par_tiling);
    
    gpu_runtime::f_vertex_set_apply_host<init_feature>(__host_edges__transposed, __host_input_f_size);
    
    int32_t vf_num_cta[2];
	int32_t vf_feat_size[2];
    vf_feat_size[0] = __host_hidden_f_size;
    vf_feat_size[1] = __host_output_f_size;
    gpu_runtime::f_vertex_set_apply_info(__host_edges__transposed, vf_num_cta[0], vf_feat_size[0]);
    gpu_runtime::f_vertex_set_apply_info(__host_edges__transposed, vf_num_cta[1], vf_feat_size[1]);
    
    float time = 0;
    int cnt = 20;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

	for (int32_t trail = 0; trail < cnt; trail++) {
        
        //layer-1
        gpu_runtime::matmul_NN(cublasHs, __device_feature_input, __device_weight_fc_1, __device_feature_input_fc_1, __host_vector_size, __host_hidden_f_size, __host_input_f_size);

        // gather
        {
            gpu_runtime::uGrapher_exec(0, lb_c, ef_num_cta, 
                __host_edges__transposed, __device_feature_input_fc_1, __device_weight_norm, __device_feature_input_out_1, 
                ef_feat_size, group_size, par_tiling);
            // cudaDeviceSynchronize();
        }

        // get_rst
        {
            gpu_runtime::f_vertex_set_apply_kernel<get_rst_1><<<vf_num_cta[0], CTA_SIZE>>>(__host_edges__transposed, vf_feat_size[0]);
            // cudaDeviceSynchronize();
        }

        ////////////////////////////////////////////////////////////////////////////////
        // layer-2
        gpu_runtime::matmul_NN(cublasHs, __device_feature_rst_1, __device_weight_fc_2, __device_feature_input_fc_2, __host_vector_size, __host_output_f_size, __host_hidden_f_size);

        // gather
        {
            gpu_runtime::uGrapher_exec(1, lb_c, ef_num_cta, 
                __host_edges__transposed, __device_feature_input_fc_2, __device_weight_norm, __device_feature_input_out_2, 
                ef_feat_size, group_size, par_tiling);
            // cudaDeviceSynchronize();
        }

        // get_rst
        {   
            gpu_runtime::f_vertex_set_apply_kernel<get_rst_2><<<vf_num_cta[1], CTA_SIZE>>>(__host_edges__transposed, vf_feat_size[1]);
            // cudaDeviceSynchronize();
	    }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    time = time / cnt;
    cout << "time: " << time << " msec" << endl;

    cuda_error = cudaGetLastError();

    if (cuda_error != cudaSuccess) {
        printf("Last CUDA Error: %s\n", cudaGetErrorString(cuda_error));
        return -1;
    }

    std::fstream fp;
	fp.open(argv[15], std::ios::out|std::ios::app);
	fp << time << std::endl;
	fp.close();

    //ready to return
    cudaFree(__device_feature_input);
    cudaFree(__device_feature_input_fc_1);
    cudaFree(__device_feature_input_out_1);
    cudaFree(__device_feature_rst_1);
    cudaFree(__device_feature_input_fc_2);
    cudaFree(__device_feature_input_out_2);
    cudaFree(__device_feature_rst_2);
    cudaFree(__device_weight_fc_1);
    cudaFree(__device_weight_fc_2);

    delete __host_feature_input;
    delete __host_feature_input_fc_1;
    delete __host_feature_input_out_1;
    delete __host_feature_rst_1;
    delete __host_feature_input_fc_2;
    delete __host_feature_input_out_2;
    delete __host_feature_rst_2;
    delete __host_weight_fc_1;
    delete __host_weight_fc_2;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    for (int i = 0; i < GPUNUM; i++) {
		cublasDestroy(cublasHs[i]);
	}

	delete cublasHs;

    return 0;
}