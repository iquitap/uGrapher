#include "gpu_intrinsics.h"
#include <cooperative_groups.h>
using namespace cooperative_groups;

#define sz_head         8

// total feature size
int32_t sz_fin;		    // (model_fin  * model_vertex)
int32_t sz_fhid;		// (model_fhid * model_vertex)

// weight size
int32_t sz_w1;       // (model_fin * model_fhid * sz_head)

int32_t sz_atten_1;  // (model_fhid * sz_head)

// edge feature size
int32_t sz_e_1;              // (model_vertex * sz_head)

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

float __device__ *feature_input;
float *__host_feature_input;
float *__device_feature_input;

float __device__ *Weight_1;
float *__host_Weight_1;
float *__device_Weight_1;

float __device__ *input_hidden_1;
float *__host_input_hidden_1;
float *__device_input_hidden_1;

float __device__ *Weight_src_1;
float *__host_Weight_src_1;
float *__device_Weight_src_1;

float __device__ *feature_el_1;
float *__host_feature_el_1;
float *__device_feature_el_1;

float __device__ *Weight_dst_1;
float *__host_Weight_dst_1;
float *__device_Weight_dst_1;

float __device__ *feature_er_1;
float *__host_feature_er_1;
float *__device_feature_er_1;

float __device__ *edge_exp_1;
float *__host_edge_exp_1;
float *__device_edge_exp_1;

void __device__ init_feature(int32_t v, int32_t f);
void __device__ init_feature(int32_t v, int32_t f) {
	int32_t f_size = input_f_size;
	feature_input[f] = (v+1 + f%f_size)/100000.0;
}

// layer1

////////////////////////////////////////////////////////////////////////////////
// rewrite u_add_v_1
void __device__ u_add_v_1(int32_t src, int32_t dst, int32_t edge, int32_t feat, float* A, float* B, float* C, int Feat_Size);
//template <typename EdgeWeightType>
void __device__ gpu_operator_body_00(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {
	// Body of the actual operator
	u_add_v_1(dst, src, edge, feat, A, B, C, Feat_Size);
}
void __device__  u_add_v_1(int32_t src, int32_t dst, int32_t edge, int32_t feat, float* A, float* B, float* C, int Feat_Size) {
    // multi-head
    C[edge * Feat_Size + feat] = A[src * Feat_Size + feat] + B[dst * Feat_Size + feat];
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

    sz_fin = __host_input_f_size * __host_vector_size;
    sz_fhid = __host_hidden_f_size * __host_vector_size;
	
	sz_w1 = __host_input_f_size * __host_hidden_f_size * sz_head;
	
	sz_atten_1 = __host_hidden_f_size * sz_head;
	
    sz_e_1 = __host_vector_size * sz_head;

    sz_edge_feat_1 = __host_edge_size * sz_head;

	cudaMalloc(&__device_feature_input, sz_fin * sizeof(float));
	cudaMemcpyToSymbol(feature_input, &__device_feature_input, sizeof(float*), 0);
	__host_feature_input = new float[sz_fin];

	cudaMalloc(&__device_Weight_1, sz_w1 * sizeof(float));
	cudaMemcpyToSymbol(Weight_1, &__device_Weight_1, sizeof(float*), 0);
	__host_Weight_1 = new float[sz_w1];

	cudaMalloc(&__device_input_hidden_1, sz_fhid * sz_head * sizeof(float));
	cudaMemcpyToSymbol(input_hidden_1, &__device_input_hidden_1, sizeof(float*), 0);
	__host_input_hidden_1 = new float[sz_fhid * sz_head];
	
    cudaMalloc(&__device_Weight_src_1, sz_atten_1 * sizeof(float));
	cudaMemcpyToSymbol(Weight_src_1, &__device_Weight_src_1, sizeof(float*), 0);
	__host_Weight_src_1 = new float[sz_atten_1];

	cudaMalloc(&__device_feature_el_1, sz_e_1 * sizeof(float));
	cudaMemcpyToSymbol(feature_el_1, &__device_feature_el_1, sizeof(float*), 0);
	__host_feature_el_1 = new float[sz_e_1];

	cudaMalloc(&__device_Weight_dst_1, sz_atten_1 * sizeof(float));
	cudaMemcpyToSymbol(Weight_dst_1, &__device_Weight_dst_1, sizeof(float*), 0);
	__host_Weight_dst_1 = new float[sz_atten_1];

	cudaMalloc(&__device_feature_er_1, sz_e_1 * sizeof(float));
	cudaMemcpyToSymbol(feature_er_1, &__device_feature_er_1, sizeof(float*), 0);
	__host_feature_er_1 = new float[sz_e_1];

	cudaMalloc(&__device_edge_exp_1, sz_edge_feat_1 * sizeof(float));
	cudaMemcpyToSymbol(edge_exp_1, &__device_edge_exp_1, sizeof(float*), 0);
	__host_edge_exp_1 = new float[sz_edge_feat_1];

    // printf("init __host_Weight_1:%d %d %d\n", __host_input_f_size, __host_hidden_f_size, __host_hidden_f_size*__host_input_f_size);
    initGPUData(__device_Weight_1, sz_w1, 0.001);
    // printf("init __host_Weight_src_1:%d %d %d\n", __host_hidden_f_size, __host_num_head, __host_num_head*__host_hidden_f_size);
    initGPUData(__device_Weight_src_1, sz_atten_1, 0.002);
    // printf("init __host_Weight_dst_1:%d %d %d\n", __host_hidden_f_size, __host_num_head, __host_num_head*__host_hidden_f_size);
    initGPUData(__device_Weight_dst_1, sz_atten_1, 0.003);

    bool run_vf = std::stoi(argv[5]);
    bool run_ef = std::stoi(argv[6]);

    // decide load balance schedule
	gpu_runtime::gnn_load_balance_type lb_c;
    int32_t ef_num_cta=0;
    int32_t ef_feat_size=sz_head;
    int32_t group_size=std::stoi(argv[9]);
    int32_t par_tiling=std::stoi(argv[10]);
    gpu_runtime::uGrapher_init<gpu_operator_body_00, gpu_operator_body_00>(
        0, argv[8], &lb_c, __host_edges__transposed, &ef_num_cta, 
        &ef_feat_size, &group_size, &par_tiling);
    
	gpu_runtime::f_vertex_set_apply_host<init_feature>(__host_edges__transposed, __host_input_f_size);
    
    float time = 0, time_vf = 0, time_ef = 0;
    float elapsed_time;
    int cnt = 20;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	for (int32_t trail = 0; trail < cnt; trail++) {

        //layer_1
        //updateVertex_z_1
        gpu_runtime::matmul_NN(cublasHs, __device_feature_input, __device_Weight_1, __device_input_hidden_1, __host_vector_size, __host_hidden_f_size * __host_num_head, __host_input_f_size);

        //updateVertex_att_src_1
        gpu_runtime::matmul_NN(cublasHs, __device_input_hidden_1, __device_Weight_src_1, __device_feature_el_1, __host_vector_size * __host_num_head, 1, __host_hidden_f_size);
        
        //updateVertex_att_dst_1
        gpu_runtime::matmul_NN(cublasHs, __device_input_hidden_1, __device_Weight_dst_1, __device_feature_er_1, __host_vector_size * __host_num_head, 1, __host_hidden_f_size);
        cudaDeviceSynchronize();

        if (run_ef) cudaEventRecord(start);
        {
			gpu_runtime::uGrapher_exec(0, &lb_c, &ef_num_cta, 
                __host_edges__transposed, __device_feature_el_1, __device_feature_er_1, __device_edge_exp_1, 
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
    CUDA_CALL(cudaFree(__device_feature_input));
    CUDA_CALL(cudaFree(__device_Weight_1));
    CUDA_CALL(cudaFree(__device_input_hidden_1));
    CUDA_CALL(cudaFree(__device_Weight_src_1));
    CUDA_CALL(cudaFree(__device_feature_el_1));
    CUDA_CALL(cudaFree(__device_feature_er_1));
    CUDA_CALL(cudaFree(__device_Weight_dst_1));
    CUDA_CALL(cudaFree(__device_edge_exp_1));

    delete __host_feature_input;
    delete __host_Weight_1;
    delete __host_input_hidden_1;
    delete __host_Weight_src_1;
    delete __host_feature_el_1;
    delete __host_feature_er_1;
    delete __host_Weight_dst_1;
    delete __host_edge_exp_1;

    // cublas free
    for (int i = 0; i < GPUNUM; i++) {
		cublasDestroy(cublasHs[i]);
	}

	delete cublasHs;
}