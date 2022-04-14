#include "gpu_intrinsics.h"
#include <cooperative_groups.h>
using namespace cooperative_groups;

#define sz_head         8

// edge feature size
int32_t sz_edge_feat_1;      // (model_edge * sz_head)

// leaky_relu_param
#define negative_slope  0.01

int32_t __delta_param;
gpu_runtime::GraphT __device__ edges;
gpu_runtime::GraphT __host_edges;
gpu_runtime::GraphT __device__ edges__transposed;
gpu_runtime::GraphT __host_edges__transposed;

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

int32_t __device__ num_head; 
int32_t __host_num_head;

float __device__ *edge_exp_1;
float *__host_edge_exp_1;
float *__device_edge_exp_1;

// layer1

////////////////////////////////////////////////////////////////////////////////
// rewrite u_add_v_1
void __device__ expf_1(int32_t edge, int32_t feat, float* A, float* C, int Feat_Size);
//template <typename EdgeWeightType>
void __device__ gpu_operator_body_02(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {
    // Body of the actual operator
    expf_1(edge, feat, A, C, Feat_Size);
}
void __device__ expf_1(int32_t edge, int32_t feat, float* A, float* C, int Feat_Size) {
    // multi-head
    C[edge * Feat_Size + feat] = __expf(A[edge * Feat_Size + feat]);
}
// rewrite u_add_v_1 end
////////////////////////////////////////////////////////////////////////////////

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
	__host_num_head = sz_head;
	cudaMemcpyToSymbol(num_head, &__host_num_head, sizeof(int32_t), 0, cudaMemcpyHostToDevice);

    sz_edge_feat_1 = __host_edge_size * sz_head;

	cudaMalloc(&__device_edge_exp_1, sz_edge_feat_1 * sizeof(float));
	cudaMemcpyToSymbol(edge_exp_1, &__device_edge_exp_1, sizeof(float*), 0);
	__host_edge_exp_1 = new float[sz_edge_feat_1];
    
    initGPUData(__device_edge_exp_1, sz_edge_feat_1, 0.001);

    bool run_vf = std::stoi(argv[5]);
    bool run_ef = std::stoi(argv[6]);

    // decide load balance schedule
	gpu_runtime::gnn_load_balance_type lb_c;
    int32_t ef_num_cta=0;
    int32_t ef_feat_size=sz_head;
    int32_t group_size=std::stoi(argv[9]);
    int32_t par_tiling=std::stoi(argv[10]);
    gpu_runtime::uGrapher_init<gpu_operator_body_02, gpu_operator_body_02>(
        0, argv[8], &lb_c, __host_edges__transposed, &ef_num_cta, 
        &ef_feat_size, &group_size, &par_tiling);
    
    float time = 0, time_vf = 0, time_ef = 0;
    float elapsed_time;
    int cnt = 20;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	for (int32_t trail = 0; trail < cnt; trail++) {

        //layer_1
        if (run_ef) cudaEventRecord(start);
        {
			gpu_runtime::uGrapher_exec(0, &lb_c, &ef_num_cta, 
                __host_edges__transposed, __device_edge_exp_1, 0, __device_edge_exp_1, 
                &ef_feat_size, &group_size, &par_tiling);
            // cudaDeviceSynchronize();
		}
        if (run_ef) {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);
		    time_ef = time_ef + elapsed_time;
        }
	}
    
    time_vf = time_vf * 1000 / cnt;
    time_ef = time_ef * 1000 / cnt;
    time = time_vf * run_vf + time_ef * run_ef;
    cout << "time: " << time << " msec" << endl;

    cuda_error = cudaGetLastError();

    if (cuda_error != cudaSuccess) {
        printf(" 600 CUDA Error: %s\n", cudaGetErrorString(cuda_error));
        return 0;
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

    //free
    CUDA_CALL(cudaFree(__device_edge_exp_1));

    delete __host_edge_exp_1;

    // cublas free
    for (int i = 0; i < GPUNUM; i++) {
		cublasDestroy(cublasHs[i]);
	}

	delete cublasHs;
}