// full-batch
// cora
// aggregation op = sum/pool/lstm/mean
// this file is for mean aggregation
#include "gpu_intrinsics.h"
#include <cooperative_groups.h>
using namespace cooperative_groups;

// total feature size
int32_t sz_fin;		    // (model_fin  * model_vertex)
int32_t sz_fhid;		// (model_fhid * model_vertex)

// weight size
int32_t sz_w1;          // (model_fin * model_fhid)

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

float __device__ *feature_input_fc_1;
float *__host_feature_input_fc_1;
float *__device_feature_input_fc_1;

float __device__ *feature_input_fc_ngh_1;
float *__host_feature_input_fc_ngh_1;
float *__device_feature_input_fc_ngh_1;

float __device__ *feature_rst_1;
float *__host_feature_rst_1;
float *__device_feature_rst_1;

float __device__ *weight_fc_1;
float *__host_weight_fc_1;
float *__device_weight_fc_1;

float __device__ *weight_fc_ngh_1;
float *__host_weight_fc_ngh_1;
float *__device_weight_fc_ngh_1;

void __device__ sum_1(gpu_runtime::GraphT graph, int32_t src, int32_t dst, int32_t feat, float* A, float* C, int Feat_Size);
void __device__ sum_1_nAtm(gpu_runtime::GraphT graph, int32_t src, int32_t dst, int32_t feat, float* A, float* C, int Feat_Size);
void __device__ get_rst_11(int32_t v, int32_t f);
void __device__ get_rst_12(int32_t v, int32_t f);
void __device__ init_feature(int32_t v, int32_t f);

//template <typename EdgeWeightType>
void __device__ gpu_operator_body_0(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {
	// Body of the actual operator
	sum_1(graph, dst, src, feat, A, C, Feat_Size);
}

//template <typename EdgeWeightType>
void __device__ gpu_operator_body_0_nAtm(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {
	// Body of the actual operator
	sum_1_nAtm(graph, dst, src, feat, A, C, Feat_Size);
}

// template<typename EdgeWeightType>
void __device__ sum_1(gpu_runtime::GraphT graph, int32_t src, int32_t dst, int32_t feat, float* A, float* C, int Feat_Size) {
    gpu_runtime::writeAdd(&C[((dst * Feat_Size) + feat)], A[((src * Feat_Size) + feat)] / graph.d_get_degree(dst));
}

// template<typename EdgeWeightType>
void __device__ sum_1_nAtm(gpu_runtime::GraphT graph, int32_t src, int32_t dst, int32_t feat, float* A, float* C, int Feat_Size) {
    C[((dst * Feat_Size) + feat)] += A[((src * Feat_Size) + feat)] / graph.d_get_degree(dst);
}

void __device__ init_feature(int32_t v, int32_t f) {
	int32_t f_size = input_f_size;
	feature_input[f] = (v + f%f_size + 1)/100000.0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// rewrite get_rst_1

void __device__ get_rst_11(int32_t v, int32_t f) {
    feature_rst_1[f] = feature_input_fc_1[f] + feature_input_out_1[f];
}

void __device__ get_rst_12(int32_t v, int32_t f) {
    float tmp = feature_rst_1[f];
    feature_rst_1[f] = tmp > 0 ? tmp : 0; // get_rst_1 have relu activation
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
        printf("192 CUDA Error: %s\n", cudaGetErrorString(cuda_error));
    }

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
    
    sz_w1 = __host_input_f_size * __host_hidden_f_size;
    
    //layer-1
    cudaMalloc(&__device_feature_input, sz_fin * sizeof(float));
    cudaMemset(__device_feature_input, 0, sz_fin * sizeof(float));
    cudaMemcpyToSymbol(feature_input, &__device_feature_input, sizeof(float*), 0);
	__host_feature_input = new float[sz_fin];
	
    cudaMalloc(&__device_feature_input_out_1, sz_fhid * sizeof(float));
    cudaMemset(__device_feature_input_out_1, 0, sz_fhid * sizeof(float));
	cudaMemcpyToSymbol(feature_input_out_1, &__device_feature_input_out_1, sizeof(float*), 0);
	__host_feature_input_out_1 = new float[sz_fhid];

    cudaMalloc(&__device_feature_input_fc_1, sz_fhid * sizeof(float));
    cudaMemset(__device_feature_input_fc_1, 0, sz_fhid * sizeof(float));
    cudaMemcpyToSymbol(feature_input_fc_1, &__device_feature_input_fc_1, sizeof(float*), 0);
    __host_feature_input_fc_1 = new float[sz_fhid];

    cudaMalloc(&__device_feature_input_fc_ngh_1, sz_fhid * sizeof(float));
    cudaMemset(__device_feature_input_fc_ngh_1, 0, sz_fhid * sizeof(float));
    cudaMemcpyToSymbol(feature_input_fc_ngh_1, &__device_feature_input_fc_ngh_1, sizeof(float*), 0);
    __host_feature_input_fc_ngh_1 = new float[sz_fhid];

    cudaMalloc(&__device_weight_fc_1, sz_w1 * sizeof(float));
    cudaMemset(__device_weight_fc_1, 0, sz_w1 * sizeof(float));
    cudaMemcpyToSymbol(weight_fc_1, &__device_weight_fc_1, sizeof(float*), 0);
    __host_weight_fc_1 = new float[sz_w1];

    cudaMalloc(&__device_weight_fc_ngh_1, sz_w1 * sizeof(float));
    cudaMemset(__device_weight_fc_ngh_1, 0, sz_w1 * sizeof(float));
    cudaMemcpyToSymbol(weight_fc_ngh_1, &__device_weight_fc_ngh_1, sizeof(float*), 0);
    __host_weight_fc_ngh_1 = new float[sz_w1];

    cudaMalloc(&__device_feature_rst_1, sz_fhid * sizeof(float));
    cudaMemset(__device_feature_rst_1, 0, sz_fhid * sizeof(float));
    cudaMemcpyToSymbol(feature_rst_1, &__device_feature_rst_1, sizeof(float*), 0);
    __host_feature_rst_1 = new float[sz_fhid];

    // printf("init __host_weight_fc_1:%d %d %d\n", __host_input_f_size, __host_hidden_f_size, __host_hidden_f_size*__host_input_f_size);
    initGPUData(__device_weight_fc_1, sz_w1, 0.001);

    // printf("init __host_weight_fc_ngh_1:%d %d %d\n", __host_input_f_size, __host_hidden_f_size, __host_hidden_f_size*__host_input_f_size);
    initGPUData(__device_weight_fc_ngh_1, sz_w1, 0.002);

    cuda_error = cudaGetLastError();

    if (cuda_error != cudaSuccess) {
        printf("300 CUDA Error: %s\n", cudaGetErrorString(cuda_error));
    }

    bool run_vf = std::stoi(argv[5]);
    bool run_ef = std::stoi(argv[6]);

    // decide load balance schedule
	gpu_runtime::gnn_load_balance_type lb_c;
    int32_t ef_num_cta=0;
    int32_t ef_feat_size=__host_hidden_f_size;
    int32_t group_size=std::stoi(argv[9]);
    int32_t par_tiling=std::stoi(argv[10]);
    gpu_runtime::uGrapher_init<gpu_operator_body_0, gpu_operator_body_0_nAtm>(
        0, argv[8], &lb_c, __host_edges__transposed, &ef_num_cta, 
        &ef_feat_size, &group_size, &par_tiling);
    
    gpu_runtime::f_vertex_set_apply_host<init_feature>(__host_edges__transposed, __host_input_f_size);
    
    int32_t vf_num_cta[2];
	int32_t vf_feat_size[2];
    vf_feat_size[0] = __host_hidden_f_size;
    vf_feat_size[1] = __host_hidden_f_size;
    gpu_runtime::f_vertex_set_apply_info(__host_edges__transposed, vf_num_cta[0], vf_feat_size[0]);
    gpu_runtime::f_vertex_set_apply_info(__host_edges__transposed, vf_num_cta[1], vf_feat_size[1]);

    float time = 0, time_vf = 0, time_ef = 0;
    float elapsed_time;
    int cnt = 20;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	for (int32_t trail = 0; trail < cnt; trail++) {
        
        // layer-1

        // matmul
        gpu_runtime::matmul_NN(cublasHs, __device_feature_input, __device_weight_fc_1, __device_feature_input_fc_1, __host_vector_size, __host_hidden_f_size, __host_input_f_size);

        gpu_runtime::matmul_NN(cublasHs, __device_feature_input, __device_weight_fc_ngh_1, __device_feature_input_fc_ngh_1, __host_vector_size, __host_hidden_f_size, __host_input_f_size);
        cudaDeviceSynchronize();

        // gather
        if (run_ef) cudaEventRecord(start);
        {
            gpu_runtime::uGrapher_exec(0, &lb_c, &ef_num_cta, 
                __host_edges__transposed, __device_feature_input_fc_ngh_1, 0, __device_feature_input_out_1, 
                &ef_feat_size, &group_size, &par_tiling);
            // cudaDeviceSynchronize();
        }
        if (run_ef) {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);
		    time_ef = time_ef + elapsed_time;
        }

        // get_rst
        if (run_vf) cudaEventRecord(start);
        {
            gpu_runtime::f_vertex_set_apply_kernel<get_rst_11><<<vf_num_cta[0], CTA_SIZE>>>(__host_edges__transposed, vf_feat_size[0]);
            // cudaDeviceSynchronize();
        }
        if (run_vf) {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);
		    time_vf = time_vf + elapsed_time;
        }
        if (run_vf) cudaEventRecord(start);
        {
            gpu_runtime::f_vertex_set_apply_kernel<get_rst_12><<<vf_num_cta[1], CTA_SIZE>>>(__host_edges__transposed, vf_feat_size[1]);
            // cudaDeviceSynchronize();
        }
        if (run_vf) {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);
		    time_vf = time_vf + elapsed_time;
        }
    }

    time_vf = time_vf * 1000 / cnt;
    time_ef = time_ef * 1000 / cnt;
    time = time_vf * run_vf + time_ef * run_ef;
    cout << "time: " << time << " msec" << endl;

	// std::cout << (all_time / ((float)1000) * 1000) << std::endl;

    cuda_error = cudaGetLastError();

    if (cuda_error != cudaSuccess) {
        printf("Last CUDA Error: %s\n", cudaGetErrorString(cuda_error));
    }

    std::fstream fp;
	if (run_vf) {
        fp.open(argv[11], std::ios::out|std::ios::app);
        fp << time_vf;
        fp.close();
    }
    if (run_ef) {
        fp.open(argv[11], std::ios::out|std::ios::app);
        fp << time_ef;
        fp.close();
    }

    //ready to return
    cudaFree(__device_feature_input);
    cudaFree(__device_feature_input_fc_1);
    cudaFree(__device_feature_input_out_1);
    cudaFree(__device_feature_input_fc_ngh_1);
    cudaFree(__device_feature_rst_1);
    cudaFree(__device_weight_fc_1);
    cudaFree(__device_weight_fc_ngh_1);

    delete __host_feature_input;
    delete __host_feature_input_fc_1;
    delete __host_feature_input_out_1;
    delete __host_feature_input_fc_ngh_1;
    delete __host_feature_rst_1;
    delete __host_weight_fc_1;
    delete __host_weight_fc_ngh_1;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    for (int i = 0; i < GPUNUM; i++) {
        cublasDestroy(cublasHs[i]);
    }

    delete cublasHs;

    return 0;
}