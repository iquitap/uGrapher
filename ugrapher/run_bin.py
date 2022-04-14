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
from util import GLOBAL

parser = argparse.ArgumentParser()
parser.add_argument("--algo", default='gcn', help="algorithm name", type=str)
parser.add_argument("--dataset", default='cora', help="dataset name", type=str)
parser.add_argument("--input_dir", default="./input/", help="input dir", type=str)
parser.add_argument("--output_dir", default="./output/", help="output dir", type=str)
parser.add_argument("--runtime_path", default="", help="output dir", type=str)
parser.add_argument("--device", default=0, help="gpu device", type=int)
parser.add_argument("--input_f", default=0, help="input feature size", type=int)
parser.add_argument("--hidden_f", default=0, help="hidden feature size", type=int)
parser.add_argument("--output_f", default=0, help="output feature size", type=int)
parser.add_argument("--run_vf", default=1, help="whether to run vertex functions", type=int)
parser.add_argument("--run_ef", default=1, help="whether to run edge functions", type=int)
parser.add_argument("--schedule", nargs="+", default=["t_vertex_group_tiling"], help="gnn load balance schedules", type=str)
parser.add_argument("--group_size", nargs="+", default=[1], help="group size parameters for edge group load balance", type=int)
parser.add_argument("--par_tiling", nargs="+", default=[1], help="parameters for tiling feature dimension", type=int)
parser.add_argument("--fine_vertex_apply", nargs="+", default=[1], help="whether to use fine grain vertex apply function, 0 for coarse grain, 1 for fine grain", type=int)

args = parser.parse_args()

output_dir = args.output_dir
algo = args.algo
dataset = args.dataset

device_command = "CUDA_VISIBLE_DEVICES=" + str(args.device) + " "
bin_output_file = output_dir + algo + ".gt.cu.o "

dataset_path = GLOBAL.dataset_path(dataset)

command = device_command + bin_output_file \
                            + dataset_path \
                            + str(args.input_f) + " " \
                            + str(args.hidden_f) + " " \
                            + str(args.output_f) + " " \
                            + str(args.run_vf) + " " \
                            + str(args.run_ef)

for i in range(len(args.schedule)):
    command = command + " " + str(args.fine_vertex_apply[i])
    command = command + " " + args.schedule[i]
    command = command + " " + str(args.group_size[i])
    command = command + " " + str(args.par_tiling[i])
    
command = command + " " + args.runtime_path

print(command)
os.system(command)
