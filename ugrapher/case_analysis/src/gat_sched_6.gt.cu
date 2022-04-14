#include "gpu_intrinsics.h"
#include <cooperative_groups.h>
using namespace cooperative_groups;

#define sz_head         8

// total feature size
int32_t sz_fhid;		// (model_fhid * model_vertex)
int32_t sz_fout;		// (model_fout * model_vertex)

// weight size
int32_t sz_w2;       // (model_fhid * model_fout * sz_head)

int32_t sz_atten_2;  // (model_fout * sz_head)

// edge feature size
int32_t sz_e_2;              // (model_vertex)

int32_t sz_edge_feat_2;      // (model_edge)

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

float __device__ *rst_1;
float *__host_rst_1;
float *__device_rst_1;

float __device__ *Weight_2;
float *__host_Weight_2;
float *__device_Weight_2;

float __device__ *input_hidden_2;
float *__host_input_hidden_2;
float *__device_input_hidden_2;

float __device__ *Weight_src_2;
float *__host_Weight_src_2;
float *__device_Weight_src_2;

float __device__ *feature_el_2;
float *__host_feature_el_2;
float *__device_feature_el_2;

float __device__ *Weight_dst_2;
float *__host_Weight_dst_2;
float *__device_Weight_dst_2;

float __device__ *feature_er_2;
float *__host_feature_er_2;
float *__device_feature_er_2;

float __device__ *edge_exp_2;
float *__host_edge_exp_2;
float *__device_edge_exp_2;

float __device__ *edge_softmax_sum_2;
float *__host_edge_softmax_sum_2;
float *__device_edge_softmax_sum_2;

float __device__ *rst_2;
float *__host_rst_2;
float *__device_rst_2;

void __device__ init_feature(int32_t v, int32_t f);
void __device__ init_feature(int32_t v, int32_t f) {
	int32_t f_size = hidden_f_size * sz_head;
	rst_1[f] = (v+1 + f%f_size)/100000.0;
}

//layer-2

////////////////////////////////////////////////////////////////////////////////
// rewrite u_add_v_2
void __device__ u_add_v_2(int32_t src, int32_t dst, int32_t edge, float* A, float* B, float* C);
//template <typename EdgeWeightType>
void __device__ gpu_operator_body_30(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {
    // Body of the actual operator
    u_add_v_2(dst, src, edge, A, B, C);
}
void __device__ u_add_v_2(int32_t src, int32_t dst, int32_t edge, float* A, float* B, float* C) {
    C[edge] = (A[src] + B[dst]);
}

void __device__ leak_relu_2(int32_t edge, float* A, float* C);
//template <typename EdgeWeightType>
void __device__ gpu_operator_body_31(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {
    // Body of the actual operator
    leak_relu_2(edge, A, C);
}
void __device__ leak_relu_2(int32_t edge, float* A, float* C) {
    float tmp = A[edge];
    C[edge] = (tmp > 0.0) ? tmp : negative_slope * tmp;
}

void __device__ expf_2(int32_t edge, float* A, float* C);
//template <typename EdgeWeightType>
void __device__ gpu_operator_body_32(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {
    // Body of the actual operator
    expf_2(edge, A, C);
}
void __device__ expf_2(int32_t edge, float* A, float* C) {
    float tmp = A[edge];
    C[edge] = __expf(tmp);
}
// rewrite u_add_v_e end
////////////////////////////////////////////////////////////////////////////////

void __device__ soft_max_rst_2(int32_t dst, int32_t edge, float* A, float* C);
//template <typename EdgeWeightType>
void __device__ gpu_operator_body_4(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {
    // Body of the actual operator
    soft_max_rst_2(src, edge, A, C);
}
void __device__  soft_max_rst_2(int32_t dst, int32_t edge, float* A, float* C) {
    gpu_runtime::writeAdd(&C[dst], A[edge]);
}

////////////////////////////////////////////////////////////////////////////////
// rewrite get_rst_2
void __device__ fdiv2(int32_t dst, int32_t edge, float* A, float* B, float* C);
//template <typename EdgeWeightType>
void __device__ gpu_operator_body_50(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {
    // Body of the actual operator
    fdiv2(src, edge, A, B, C);
}
void __device__ fdiv2(int32_t dst, int32_t edge, float* A, float* B, float* C) {
    float tmp = A[edge];
    C[edge] = __fdividef(tmp, B[dst]);
}

void __device__ get_rst_2(int32_t src, int32_t dst, int32_t edge, int32_t feat, float* A, float* B, float* C, int32_t Feat_Size);
//template <typename EdgeWeightType>
void __device__ gpu_operator_body_51(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {
    // Body of the actual operator
    get_rst_2(dst, src, edge, feat, A, B, C, Feat_Size);
}
void __device__  get_rst_2(int32_t src, int32_t dst, int32_t edge, int32_t feat, float* A, float* B, float* C, int32_t Feat_Size) {
    float tmp = A[edge];
    gpu_runtime::writeAdd(&C[dst * Feat_Size + feat], tmp * B[src * Feat_Size + feat]);
}

void __device__ get_rst_2_nAtm(int32_t src, int32_t dst, int32_t edge, int32_t feat, float* A, float* B, float* C, int32_t Feat_Size);
//template <typename EdgeWeightType>
void __device__ gpu_operator_body_51_nAtm(gpu_runtime::GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {
    // Body of the actual operator
    get_rst_2_nAtm(dst, src, edge, feat, A, B, C, Feat_Size);
}
void __device__  get_rst_2_nAtm(int32_t src, int32_t dst, int32_t edge, int32_t feat, float* A, float* B, float* C, int32_t Feat_Size) {
    float tmp = A[edge];
    C[dst * Feat_Size + feat] += tmp * B[src * Feat_Size + feat];
}

////////////////////////////////////////////////////////////////////////////////

// ELU_2
void __device__ elu_2(int32_t v, int32_t f) {
    if(rst_2[f] < 0) {
        rst_2[f] = __expf(rst_2[f]) - 1;
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
    sz_fout = __host_output_f_size * __host_vector_size;
	
    sz_w2 = __host_hidden_f_size * __host_output_f_size * sz_head;
	
    sz_atten_2 = __host_output_f_size * sz_head;
	
    sz_e_2 = __host_vector_size;

    sz_edge_feat_2 = __host_edge_size;

	cudaMalloc(&__device_rst_1, sz_fhid * sz_head * sizeof(float));
	cudaMemcpyToSymbol(rst_1, &__device_rst_1, sizeof(float*), 0);
	__host_rst_1 = new float[sz_fhid * sz_head];

	cudaMalloc(&__device_Weight_2, sz_w2 * sizeof(float));
	cudaMemcpyToSymbol(Weight_2, &__device_Weight_2, sizeof(float*), 0);
	__host_Weight_2 = new float[sz_w2];

	cudaMalloc(&__device_input_hidden_2, sz_fout * sizeof(float));
	cudaMemcpyToSymbol(input_hidden_2, &__device_input_hidden_2, sizeof(float*), 0);
	__host_input_hidden_2 = new float[sz_fout];
	
    cudaMalloc(&__device_Weight_src_2, sz_atten_2 * sizeof(float));
	cudaMemcpyToSymbol(Weight_src_2, &__device_Weight_src_2, sizeof(float*), 0);
	__host_Weight_src_2 = new float[sz_atten_2];
	
    cudaMalloc(&__device_feature_el_2, sz_e_2 * sizeof(float));
	cudaMemcpyToSymbol(feature_el_2, &__device_feature_el_2, sizeof(float*), 0);
	__host_feature_el_2 = new float[sz_e_2];
	
    cudaMalloc(&__device_Weight_dst_2, sz_atten_2 * sizeof(float));
	cudaMemcpyToSymbol(Weight_dst_2, &__device_Weight_dst_2, sizeof(float*), 0);
	__host_Weight_dst_2 = new float[sz_atten_2];

	cudaMalloc(&__device_feature_er_2, sz_e_2 * sizeof(float));
	cudaMemcpyToSymbol(feature_er_2, &__device_feature_er_2, sizeof(float*), 0);
	__host_feature_er_2 = new float[sz_e_2];
	
    cudaMalloc(&__device_edge_exp_2, sz_edge_feat_2 * sizeof(float));
	cudaMemcpyToSymbol(edge_exp_2, &__device_edge_exp_2, sizeof(float*), 0);
	__host_edge_exp_2 = new float[sz_edge_feat_2];

	cudaMalloc(&__device_edge_softmax_sum_2, sz_e_2 * sizeof(float));
	cudaMemcpyToSymbol(edge_softmax_sum_2, &__device_edge_softmax_sum_2, sizeof(float*), 0);
	__host_edge_softmax_sum_2 = new float[sz_e_2];
	
    cudaMalloc(&__device_rst_2, sz_fout * sizeof(float));
	cudaMemcpyToSymbol(rst_2, &__device_rst_2, sizeof(float*), 0);
	__host_rst_2 = new float[sz_fout];

	gpu_runtime::f_vertex_set_apply_host<init_feature>(__host_edges__transposed, __host_hidden_f_size * sz_head);

    // printf("init __host_Weight_2:%d %d %d\n", __host_hidden_f_size, __host_output_f_size, __host_output_f_size*__host_hidden_f_size);
    initGPUData(__device_Weight_2, sz_w2, 0.004);
    // printf("init __host_Weight_src_2:%d %d %d\n", __host_output_f_size, __host_num_head, __host_num_head*__host_output_f_size);
    initGPUData(__device_Weight_src_2, sz_atten_2, 0.005);
    // printf("init __host_Weight_dst_2:%d %d %d\n", __host_output_f_size, __host_num_head, __host_num_head*__host_output_f_size);
    initGPUData(__device_Weight_dst_2, sz_atten_2, 0.006);

    bool run_vf = std::stoi(argv[5]);
    bool run_ef = std::stoi(argv[6]);

    // decide load balance schedule
	gpu_runtime::gnn_load_balance_type lb_c[6];
    int32_t ef_num_cta[6]={0,0,0,0,0,0};
    int32_t ef_feat_size[6]={1,1,1,1,1,__host_output_f_size};
    int32_t group_size[6]={1,1,1,1,1,std::stoi(argv[9])};
    int32_t par_tiling[6]={1,1,1,1,1,std::stoi(argv[10])};
    gpu_runtime::uGrapher_init<gpu_operator_body_30, gpu_operator_body_30>(
        0, "t_edge_group_tiling", lb_c, __host_edges__transposed, ef_num_cta, 
        ef_feat_size, group_size, par_tiling);
    gpu_runtime::uGrapher_init<gpu_operator_body_31, gpu_operator_body_31>(
        1, "t_edge_group_tiling", lb_c, __host_edges__transposed, ef_num_cta, 
        ef_feat_size, group_size, par_tiling);
    gpu_runtime::uGrapher_init<gpu_operator_body_32, gpu_operator_body_32>(
        2, "t_edge_group_tiling", lb_c, __host_edges__transposed, ef_num_cta, 
        ef_feat_size, group_size, par_tiling);
    gpu_runtime::uGrapher_init<gpu_operator_body_4, gpu_operator_body_4>(
        3, "t_edge_group_tiling", lb_c, __host_edges__transposed, ef_num_cta, 
        ef_feat_size, group_size, par_tiling);
    gpu_runtime::uGrapher_init<gpu_operator_body_50, gpu_operator_body_50>(
        4, "t_edge_group_tiling", lb_c, __host_edges__transposed, ef_num_cta, 
        ef_feat_size, group_size, par_tiling);
    gpu_runtime::uGrapher_init<gpu_operator_body_51, gpu_operator_body_51_nAtm>(
        5, argv[8], lb_c, __host_edges__transposed, ef_num_cta, 
        ef_feat_size, group_size, par_tiling);
    
    int32_t vf_num_cta;
    int32_t vf_feat_size;
    vf_feat_size = __host_output_f_size;
    gpu_runtime::f_vertex_set_apply_info(__host_edges__transposed, vf_num_cta, vf_feat_size);
        
    float time = 0, time_vf = 0, time_ef = 0;
    float elapsed_time;
    int cnt = 20;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	for (int32_t trail = 0; trail < cnt; trail++) {

        //layer_2
        // updateVertex_z_2
        gpu_runtime::matmul_NN(cublasHs, __device_rst_1, __device_Weight_2, __device_input_hidden_2, __host_vector_size, __host_output_f_size, __host_hidden_f_size * sz_head);

        //updateVertex_att_src_2
        gpu_runtime::matmul_NN(cublasHs, __device_input_hidden_2, __device_Weight_src_2, __device_feature_el_2, __host_vector_size, 1, __host_output_f_size);

        //updateVertex_att_dst_2
        gpu_runtime::matmul_NN(cublasHs, __device_input_hidden_2, __device_Weight_dst_2, __device_feature_er_2, __host_vector_size, 1, __host_output_f_size);
        cudaDeviceSynchronize();

        {
            gpu_runtime::uGrapher_exec(0, lb_c, ef_num_cta, 
                __host_edges__transposed, __device_feature_el_2, __device_feature_er_2, __device_edge_exp_2, 
                ef_feat_size, group_size, par_tiling);
            cudaDeviceSynchronize();
        }
        {
            gpu_runtime::uGrapher_exec(1, lb_c, ef_num_cta, 
                __host_edges__transposed, __device_edge_exp_2, 0, __device_edge_exp_2, 
                ef_feat_size, group_size, par_tiling);
            cudaDeviceSynchronize();
        }
        {
            gpu_runtime::uGrapher_exec(2, lb_c, ef_num_cta, 
                __host_edges__transposed, __device_edge_exp_2, 0, __device_edge_exp_2, 
                ef_feat_size, group_size, par_tiling);
            cudaDeviceSynchronize();
        }

        // exp
        {
            gpu_runtime::uGrapher_exec(3, lb_c, ef_num_cta, 
                __host_edges__transposed, __device_edge_exp_2, 0, __device_edge_softmax_sum_2, 
                ef_feat_size, group_size, par_tiling);
            cudaDeviceSynchronize();
        }

        {
            gpu_runtime::uGrapher_exec(4, lb_c, ef_num_cta, 
                __host_edges__transposed, __device_edge_exp_2, __device_edge_softmax_sum_2, __device_edge_exp_2, 
                ef_feat_size, group_size, par_tiling);
            cudaDeviceSynchronize();
        }

        if (run_ef) cudaEventRecord(start);
        {
			gpu_runtime::uGrapher_exec(5, lb_c, ef_num_cta, 
                __host_edges__transposed, __device_edge_exp_2, __device_input_hidden_2, __device_rst_2, 
                ef_feat_size, group_size, par_tiling);
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
            gpu_runtime::f_vertex_set_apply_kernel<elu_2><<<vf_num_cta, CTA_SIZE>>>(__host_edges__transposed, vf_feat_size);
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
    CUDA_CALL(cudaFree(__device_rst_1));

    CUDA_CALL(cudaFree(__device_Weight_2));
    CUDA_CALL(cudaFree(__device_input_hidden_2));
    CUDA_CALL(cudaFree(__device_Weight_src_2));
    CUDA_CALL(cudaFree(__device_feature_el_2));
    CUDA_CALL(cudaFree(__device_feature_er_2));
    CUDA_CALL(cudaFree(__device_Weight_dst_2));
    CUDA_CALL(cudaFree(__device_edge_exp_2));
    CUDA_CALL(cudaFree(__device_edge_softmax_sum_2));
    CUDA_CALL(cudaFree(__device_rst_2));

    delete __host_rst_1;

    delete __host_Weight_2;
    delete __host_input_hidden_2;
    delete __host_Weight_src_2;
    delete __host_feature_el_2;
    delete __host_feature_er_2;
    delete __host_Weight_dst_2;
    delete __host_edge_exp_2;
    delete __host_edge_softmax_sum_2;
    delete __host_rst_2;

    // cublas free
    for (int i = 0; i < GPUNUM; i++) {
		cublasDestroy(cublasHs[i]);
	}

	delete cublasHs;
}