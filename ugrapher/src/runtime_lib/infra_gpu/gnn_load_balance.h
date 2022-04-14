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

#ifndef GNN_GRAPHIT_GPU_LOAD_BALANCE_H
#define GNN_GRAPHIT_GPU_LOAD_BALANCE_H

#include "infra_gpu/graph.h"
#include <cooperative_groups.h>
using namespace cooperative_groups;

#define WARP_SIZE 32

namespace gpu_runtime {

using gnn_vertex_func_type = void (int32_t, int32_t, int32_t);


using gnn_load_balance_payload_type = void (GraphT, int32_t, int32_t, int32_t, int32_t, float*, float*, float*, int);

template < void body(int32_t, int32_t)>
static void __global__ f_vertex_set_apply_kernel(GraphT graph, int32_t par_feat_size) {
	int32_t thread_id = threadIdx.x + blockDim.x * blockIdx.x;
	int32_t vid = thread_id / par_feat_size;
	if (vid >= graph.num_vertices) return;
	// int32_t vidx = AccessorType::getElement(frontier, vid);
	body(vid, thread_id);
}
//template <typename EdgeWeightType>
void __host__ f_vertex_set_apply_info(GraphT &graph, int32_t& num_cta, int32_t par_feat_size) {
	int32_t num_threads = graph.num_vertices * par_feat_size;
	num_cta = (num_threads + CTA_SIZE - 1) / CTA_SIZE;
}

template < void body(int32_t, int32_t)>
void __host__ f_vertex_set_apply_host(GraphT &graph, int32_t feature_size) {
	int32_t num_cta;

	f_vertex_set_apply_info(graph, num_cta, feature_size);
	f_vertex_set_apply_kernel< body><<<num_cta, CTA_SIZE>>>(graph, feature_size);
}


// Thread vertex group tile mapping
template < gnn_load_balance_payload_type load_balance_payload>
static void __global__ t_vertex_group_tiling_load_balance_kernel(
	GraphT graph, float* A, float* B, float* C, 
	int32_t par_feat_size, int32_t group_size, int32_t par_tiling) {

	int32_t thread_id = threadIdx.x + blockDim.x * blockIdx.x;
	int32_t tile_id = thread_id & ((1<<par_tiling)-1);

	int32_t group_start_vid = (thread_id >> par_tiling) * group_size;
	int32_t num_v = graph.num_vertices;
	if (group_start_vid >= num_v) return;

	for (int32_t src = group_start_vid; src < group_start_vid + group_size && src < num_v; ++src) {
		// int32_t src = AccessorType::getElement(input_frontier, vid);
		// if (src_filter(src) == false) break;
		for (int32_t eid = graph.d_src_offsets[src]; eid < graph.d_src_offsets[src+1]; eid++) {
			int32_t dst = graph.d_edge_dst[eid];
			// load_balance_payload(graph, src, dst, eid, tile_id, 1<<par_tiling, input_frontier, output_frontier);
			for (int32_t i = tile_id; i < par_feat_size; i += 1<<par_tiling) {
				load_balance_payload(graph, src, dst, eid, i, A, B, C, par_feat_size);
			}
		}
	}
}
//template <typename EdgeWeightType>
void __host__ t_vertex_group_tiling_load_balance_info(
	GraphT &graph, int32_t &num_cta, 
	int32_t par_feat_size, int32_t group_size, int32_t& par_tiling) {

	if (par_tiling > par_feat_size) par_tiling = par_feat_size;
	int32_t pl=0, pe=1;
	while (pe <= par_tiling) pl++, pe <<= 1;
	par_tiling = pl - 1;

	int32_t group_cnt = (graph.num_vertices + group_size - 1) / group_size;
    int32_t num_threads = group_cnt << par_tiling;
	num_cta = (num_threads + CTA_SIZE-1) / CTA_SIZE;
}

// Warp vertex group tile mapping
template < gnn_load_balance_payload_type load_balance_payload>
static void __global__ w_vertex_group_tiling_load_balance_kernel(
	GraphT graph, float* A, float* B, float* C, 
	int32_t par_feat_size, int32_t group_size, int32_t par_tiling) {

	int32_t thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    int32_t warp_id = thread_id >> 5;
    int32_t lane_id = thread_id & 31;
	int32_t tile_id = warp_id & ((1<<par_tiling)-1);
	int32_t start_feat = tile_id * WARP_SIZE + lane_id;

    if (start_feat >= par_feat_size) return;

	int32_t group_start_vid = (warp_id >> par_tiling) * group_size;
	int32_t num_v = graph.num_vertices;
	if (group_start_vid >= num_v) return;

	for (int32_t src = group_start_vid; src < group_start_vid + group_size && src < num_v; ++src) {
		// int32_t src = AccessorType::getElement(input_frontier, vid);
		// if (src_filter(src) == false) break;
		for (int32_t eid = graph.d_src_offsets[src]; eid < graph.d_src_offsets[src+1]; eid++) {
			int32_t dst = graph.d_edge_dst[eid];
			// load_balance_payload(graph, src, dst, eid, lane_id, WARP_SIZE << par_tiling, input_frontier, output_frontier);
			for (int32_t i = lane_id; i < par_feat_size; i += WARP_SIZE<<par_tiling) {
				load_balance_payload(graph, src, dst, eid, i, A, B, C, par_feat_size);
			}
		}
	}
}
//template <typename EdgeWeightType>
void __host__ w_vertex_group_tiling_load_balance_info(
	GraphT &graph, int32_t &num_cta, 
	int32_t par_feat_size, int32_t group_size, int32_t& par_tiling) {
	
	int32_t max_par_tiling = (par_feat_size + WARP_SIZE-1) / WARP_SIZE;
	if (par_tiling > max_par_tiling) par_tiling = max_par_tiling;
	int32_t pl=0, pe=1;
	while (pe <= par_tiling) pl++, pe <<= 1;
	par_tiling = pl - 1;

	int32_t group_cnt = (graph.num_vertices + group_size - 1) / group_size;
    int32_t num_threads = (group_cnt * WARP_SIZE) << par_tiling;
	num_cta = (num_threads + CTA_SIZE-1) / CTA_SIZE;
}


// EDGE_ONLY LOAD BALANCE FUNCTIONS

// Thread edge group tile mapping
template < gnn_load_balance_payload_type load_balance_payload>
static void __global__ t_edge_group_tiling_load_balance_kernel(
	GraphT graph, float* A, float* B, float* C, 
	int32_t par_feat_size, int32_t group_size, int32_t par_tiling) {

	int32_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int32_t tile_id = thread_id & ((1<<par_tiling)-1);

	int32_t group_start_eid = (thread_id >> par_tiling) * group_size;
	if (group_start_eid >= graph.num_edges) return;
	
	for (int32_t eid = group_start_eid; eid < group_start_eid + group_size && eid < graph.num_edges; eid++) {
		int32_t src = graph.d_edge_src[eid];
		// if (src_filter(src) == true) {
		int32_t dst = graph.d_edge_dst[eid];
		// load_balance_payload(graph, src, dst, e_id, tile_id, 1<<par_tiling, input_frontier, output_frontier);
		for (int32_t i = tile_id; i < par_feat_size; i += 1<<par_tiling) {
			load_balance_payload(graph, src, dst, eid, i, A, B, C, par_feat_size);
		}
		// }
	}
}
//template <typename EdgeWeightType>
void __host__ t_edge_group_tiling_load_balance_info(
	GraphT &graph, int32_t &num_cta, 
	int32_t par_feat_size, int32_t group_size, int32_t& par_tiling) {
    
	if (par_tiling > par_feat_size) par_tiling = par_feat_size;
	int32_t pl=0, pe=1;
	while (pe <= par_tiling) pl++, pe <<= 1;
	par_tiling = pl - 1;

	int32_t group_cnt = (graph.num_edges + group_size - 1) / group_size;
    int32_t num_threads = group_cnt << par_tiling;
	num_cta = (num_threads + CTA_SIZE-1) / CTA_SIZE;
}


// EDGE_GROUP LOAD BALANCE FUNCTIONS

// Warp edge group tile mapping
template < gnn_load_balance_payload_type load_balance_payload>
static void __global__ w_edge_group_tiling_load_balance_kernel(
	GraphT graph, float* A, float* B, float* C, 
	int32_t par_feat_size, int32_t group_size, int32_t par_tiling) {

	int32_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t warp_id = thread_id >> 5;
    int32_t lane_id = thread_id & 31;
	int32_t tile_id = warp_id & ((1<<par_tiling)-1);
	int32_t start_feat = tile_id * WARP_SIZE + lane_id;

    if (start_feat >= par_feat_size) return;

	int32_t group_start_eid = (warp_id >> par_tiling) * group_size;
	if (group_start_eid >= graph.num_edges) return;
	
	for (int32_t eid = group_start_eid; eid < group_start_eid + group_size && eid < graph.num_edges; eid++) {
		int32_t src = graph.d_edge_src[eid];
		// if (src_filter(src) == true) {
		int32_t dst = graph.d_edge_dst[eid];
		// load_balance_payload(graph, src, dst, e_id, start_feat, WARP_SIZE << par_tiling, input_frontier, output_frontier);
		for (int32_t i = start_feat; i < par_feat_size; i += WARP_SIZE<<par_tiling) {
			load_balance_payload(graph, src, dst, eid, i, A, B, C, par_feat_size);
		}
		// }
	}
}
//template <typename EdgeWeightType>
void __host__ w_edge_group_tiling_load_balance_info(
	GraphT &graph, int32_t &num_cta, 
	int32_t par_feat_size, int32_t group_size, int32_t& par_tiling) {
    
	int32_t max_par_tiling = (par_feat_size + WARP_SIZE-1) / WARP_SIZE;
	if (par_tiling > max_par_tiling) par_tiling = max_par_tiling;
	int32_t pl=0, pe=1;
	while (pe <= par_tiling) pl++, pe <<= 1;
	par_tiling = pl - 1;

	int32_t group_cnt = (graph.num_edges + group_size - 1) / group_size;
    int32_t num_threads = (group_cnt * WARP_SIZE) << par_tiling;
	num_cta = (num_threads + CTA_SIZE-1) / CTA_SIZE;
}


//template <typename EdgeWeightType>
void __device__ gpu_operator_body_empty(GraphT graph, 
    int32_t src, int32_t dst, int32_t edge, int32_t feat, 
    float* A, float* B, float* C, int Feat_Size) {}

typedef decltype(t_vertex_group_tiling_load_balance_kernel<gpu_operator_body_empty>)* gnn_load_balance_type;

template <
        gnn_load_balance_payload_type payload,
        gnn_load_balance_payload_type payload_nAtm>
void __host__ uGrapher_init(int idx_layer, std::string lb_name,
    gnn_load_balance_type* lb_c, GraphT &graph, int32_t* ef_num_cta, 
    int32_t* featsz, int32_t* groupsz, int32_t* tilesz) {
		
    if (lb_name == "t_vertex_group_tiling") {
        t_vertex_group_tiling_load_balance_info(
            graph, ef_num_cta[idx_layer], featsz[idx_layer], groupsz[idx_layer], tilesz[idx_layer]);
        lb_c[idx_layer] = &t_vertex_group_tiling_load_balance_kernel<payload_nAtm>;
    }
    else if (lb_name == "w_vertex_group_tiling") {
        w_vertex_group_tiling_load_balance_info(
            graph, ef_num_cta[idx_layer], featsz[idx_layer], groupsz[idx_layer], tilesz[idx_layer]);
        lb_c[idx_layer] = &w_vertex_group_tiling_load_balance_kernel<payload_nAtm>;
    }
    else if (lb_name == "t_edge_group_tiling") {
        t_edge_group_tiling_load_balance_info(
            graph, ef_num_cta[idx_layer], featsz[idx_layer], groupsz[idx_layer], tilesz[idx_layer]);
        lb_c[idx_layer] = &t_edge_group_tiling_load_balance_kernel<payload>;
    }
	else if (lb_name == "w_edge_group_tiling") {
        w_edge_group_tiling_load_balance_info(
            graph, ef_num_cta[idx_layer], featsz[idx_layer], groupsz[idx_layer], tilesz[idx_layer]);
		lb_c[idx_layer] = &w_edge_group_tiling_load_balance_kernel<payload>;
	}
}

// //template <typename EdgeWeightType>
void __host__ uGrapher_exec(int i, gnn_load_balance_type* lb_c, int32_t* num_cta, 
	GraphT& graph, float* A, float* B, float* C,
	int32_t* featsz, int32_t* groupsz, int32_t* tilesz) {
    
	lb_c[i]<<<num_cta[i],CTA_SIZE>>>(graph, A, B, C, featsz[i], groupsz[i], tilesz[i]);
}

}

#endif

