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
# parser.add_argument("--input_dir", default="./input/", help="input dir", type=str)
parser.add_argument("--gpu_type", default="V100", help="gpu type", type=str)
parser.add_argument("--output_dir", default="./output/", help="output dir", type=str)

args = parser.parse_args()


nvcc_bin = GLOBAL.nvcc_bin
gcpp_bin = GLOBAL.gcpp_bin
lib_dir = GLOBAL.lib_dir

# input_dir = args.input_dir
output_dir = args.output_dir
algo = args.algo

cuda_output_file = output_dir + algo + ".gt" + ".cu"
bin_output_file = cuda_output_file + ".o"

if args.gpu_type == "V100":
    nvcc_command = nvcc_bin + " -ccbin " + gcpp_bin + "   -lcublas -rdc=true -DCTA_SIZE=512 -gencode arch=compute_70,code=sm_70 -std=c++11 -O3 -I " + lib_dir + "  -Xcompiler \"-w\" -Wno-deprecated-gpu-targets --use_fast_math -Xptxas \" -dlcm=ca --maxrregcount=64\" " + cuda_output_file + " -o " + bin_output_file
elif args.gpu_type == "A100":
    nvcc_command = nvcc_bin + " -ccbin " + gcpp_bin + "   -lcublas -rdc=true -DCTA_SIZE=512 -gencode arch=compute_80,code=sm_80 -std=c++11 -O3 -I " + lib_dir + "  -Xcompiler \"-w\" -Wno-deprecated-gpu-targets --use_fast_math -Xptxas \" -dlcm=ca --maxrregcount=64\" " + cuda_output_file + " -o " + bin_output_file
else:
    print(f"GPU type {args.gpu_type} is not supported!")
    exit()
#os.system("rm " + bin_output_file)
print(nvcc_command)
os.system(nvcc_command)
