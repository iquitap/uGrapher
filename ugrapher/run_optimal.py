# Copyright (c) 2022, Yangjie Zhou.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import shutil
import pandas as pd
import csv
import numpy as np
from util import GLOBAL

parser = argparse.ArgumentParser()
parser.add_argument("--algo", nargs="+", default=["gat"], help="algorithm name", type=str)
parser.add_argument("--gpu_type", default="V100", help="gpu type", type=str)
parser.add_argument("--dataset", nargs="+", default=None, help="dataset name", type=str)
parser.add_argument("--device", default=0, help="gpu device", type=int)

args = parser.parse_args()

algo_info = pd.read_csv("algo_info.csv", index_col="Algorithm")

if args.dataset:
    dataset = args.dataset
else:
    dataset = GLOBAL.datasets

fhid_mp = GLOBAL.fhid_mp

src_dir = './end2end/src/'
idx_cols = ["Dataset","Algo","EF_Idx"]
opt_path = "./benchmark_opt_sched/" + args.gpu_type + "/all_optimal_sched.csv"
opt_info = pd.read_csv(opt_path, index_col=idx_cols)

for a in args.algo:
    num_sched = algo_info.loc[a]["num_schedule"]
    f_hid = fhid_mp[a]

    nvcc_cu_command = f"python nvcc_cu.py --algo {a} --gpu_type {args.gpu_type} --output_dir {src_dir}\n"
    os.system(nvcc_cu_command)

    data_info = pd.read_csv("datainfo.csv", index_col="Dataset")

    for data in dataset:
        f_in, f_out = data_info.loc[data]["input_f"], data_info.loc[data]["output_f"]

        def emit(grainset, scheduleset, groupszset, tileszset):
            run_bin_command = "python run_bin.py --algo " + a \
                            + " --output_dir " + src_dir \
                            + " --device " + str(args.device) \
                            + " --dataset " + data \
                            + " --input_f " + str(f_in) \
                            + " --hidden_f " + str(f_hid) \
                            + " --output_f " + str(f_out)

            run_bin_command = run_bin_command + " --fine_vertex_apply"
            for grain in grainset:
                run_bin_command = run_bin_command + " " + str(grain)

            run_bin_command = run_bin_command + " --schedule"
            for schedule in scheduleset:
                run_bin_command = run_bin_command + " " + schedule
            
            run_bin_command = run_bin_command + " --group_size"
            for groupsz in groupszset:
                run_bin_command = run_bin_command + " " + groupsz
            
            run_bin_command = run_bin_command + " --par_tiling"
            for tilesz in tileszset:
                run_bin_command = run_bin_command + " " + tilesz

            # run_bin_command = run_bin_command + " --runtime_path " + result_path

            print(run_bin_command)
            os.system(run_bin_command)

        scheduleset = []
        groupszset = []
        tileszset = []
        grainset = []

        for i in range(num_sched):
            r_sched = opt_info.loc[data, a, i]["Opt_schedule"]
            grainset.append(1)
            rs, rg, rt = r_sched.split("_")

            if rs=="TV":
                scheduleset.append("t_vertex_group_tiling")
            elif rs=="WV":
                scheduleset.append("w_vertex_group_tiling")
            elif rs=="TE":
                scheduleset.append("t_edge_group_tiling")
            else:
                scheduleset.append("w_edge_group_tiling")
            
            groupszset.append(rg[1:])
            tileszset.append(rt[1:])

        emit(grainset, scheduleset, groupszset, tileszset)