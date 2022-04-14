#include "gpu_intrinsics.h"
#include <cooperative_groups.h>
using namespace cooperative_groups;

#define sz_head         8

// total feature size
int32_t sz_fhid;		// (model_fhid * model_vertex)

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

float __device__ *input_hidden_1;
float *__host_input_hidden_1;
float *__device_input_hidden_1;

float __device__ *edge_exp_1;
float *__host_edge_exp_1;
float *__device_edge_exp_1;

float __device__ *rst_1;
float *__host_rst_1;
float *__device_rst_1;

void __device__ init_feature(int32_t v, int32_t f);
void __device__ init_feature(int32_t v, int32_t f) {
	int32_t f_size = hidden_f_size * sz_head;
	input_hidden_1[f] = (v+1 + f%f_size)/100000.0;
}

// layer1

////////////////////////////////////////////////////////////////////////////////
// rewrite get_rst_1
void __device__ get_rst_1(int32_t src, int32_t dst, int32_t edge, int32_t feat, float* A, float* B, float* C, int32_t Feat_Size);
//template <typename EdgeWeightType>
void __device__ gpu_operator_body_21(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {
    // Body of the actual operator
    get_rst_1(dst, src, edge, feat, A, B, C, Feat_Size);
}
void __device__  get_rst_1(int32_t src, int32_t dst, int32_t edge, int32_t feat, float* A, float* B, float* C, int32_t Feat_Size) {
    // IF fuse two loop, redundant division for tmp
    float tmp = A[edge * sz_head + feat % sz_head];
    gpu_runtime::writeAdd(&C[dst * Feat_Size + feat], tmp * B[src * Feat_Size + feat]);
}

void __device__ get_rst_1_nAtm(int32_t src, int32_t dst, int32_t edge, int32_t feat, float* A, float* B, float* C, int32_t Feat_Size);
//template <typename EdgeWeightType>
void __device__ gpu_operator_body_21_nAtm(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {
    // Body of the actual operator
    get_rst_1_nAtm(dst, src, edge, feat, A, B, C, Feat_Size);
}
void __device__  get_rst_1_nAtm(int32_t src, int32_t dst, int32_t edge, int32_t feat, float* A, float* B, float* C, int32_t Feat_Size) {
    // IF fuse two loop, redundant division for tmp
    float tmp = A[edge * sz_head + feat % sz_head];
    C[dst * Feat_Size + feat] += tmp * B[src * Feat_Size + feat];
}
// rewrite get_rst_1 end
////////////////////////////////////////////////////////////////////////////////

//vertex parallel
void __device__  elu_1(int32_t v, int32_t f) {
    if(rst_1[f] < 0) {
        rst_1[f] = __expf(rst_1[f]) - 1;
    }
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

    sz_fhid = __host_hidden_f_size * __host_vector_size;

    sz_edge_feat_1 = __host_edge_size * sz_head;

    cudaMalloc(&__device_input_hidden_1, sz_fhid * sz_head * sizeof(float));
	cudaMemcpyToSymbol(input_hidden_1, &__device_input_hidden_1, sizeof(float*), 0);
	__host_input_hidden_1 = new float[sz_fhid * sz_head];
	
	cudaMalloc(&__device_edge_exp_1, sz_edge_feat_1 * sizeof(float));
	cudaMemcpyToSymbol(edge_exp_1, &__device_edge_exp_1, sizeof(float*), 0);
	__host_edge_exp_1 = new float[sz_edge_feat_1];

	cudaMalloc(&__device_rst_1, sz_fhid * sz_head * sizeof(float));
	cudaMemcpyToSymbol(rst_1, &__device_rst_1, sizeof(float*), 0);
	__host_rst_1 = new float[sz_fhid * sz_head];

	gpu_runtime::f_vertex_set_apply_host<init_feature>(__host_edges__transposed, __host_hidden_f_size * sz_head);

    initGPUData(__device_edge_exp_1, sz_edge_feat_1, 0.001);

    bool run_vf = std::stoi(argv[5]);
    bool run_ef = std::stoi(argv[6]);

    // decide load balance schedule
	gpu_runtime::gnn_load_balance_type lb_c;
    int32_t ef_num_cta=0;
    int32_t ef_feat_size=__host_hidden_f_size*sz_head;
    int32_t group_size=std::stoi(argv[9]);
    int32_t par_tiling=std::stoi(argv[10]);
    gpu_runtime::uGrapher_init<gpu_operator_body_21, gpu_operator_body_21_nAtm>(
        0, argv[8], &lb_c, __host_edges__transposed, &ef_num_cta, 
        &ef_feat_size, &group_size, &par_tiling);
    
    int32_t vf_num_cta;
    int32_t vf_feat_size;
    vf_feat_size = __host_hidden_f_size*sz_head;
    gpu_runtime::f_vertex_set_apply_info(__host_edges__transposed, vf_num_cta, vf_feat_size);
        
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
                __host_edges__transposed, __device_edge_exp_1, __device_input_hidden_1, __device_rst_1, 
                &ef_feat_size, &group_size, &par_tiling);
            // cudaDeviceSynchronize();
		}
        if (run_ef) {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);
		    time_ef = time_ef + elapsed_time;
        }

        if (run_vf) cudaEventRecord(start);
	    {   
            gpu_runtime::f_vertex_set_apply_kernel<elu_1><<<vf_num_cta, CTA_SIZE>>>(__host_edges__transposed, vf_feat_size);
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
    CUDA_CALL(cudaFree(__device_input_hidden_1));
    CUDA_CALL(cudaFree(__device_edge_exp_1));
    CUDA_CALL(cudaFree(__device_rst_1));

    delete __host_input_hidden_1;
    delete __host_edge_exp_1;
    delete __host_rst_1;

    // cublas free
    for (int i = 0; i < GPUNUM; i++) {
		cublasDestroy(cublasHs[i]);
	}

	delete cublasHs;
}