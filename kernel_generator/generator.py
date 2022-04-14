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

class Generator:
    def __init__(self, args):
        self._edge_op = args.edge_op
        self._gather_op = args.gather_op
        self._a_t = args.A_T
        self._b_t = args.B_T
        self._c_t = args.C_T

        self._output_dir = args.output_dir
        self._output_name = args.output_file_name

        self._input_dir = args.input_dir
        self._input_header = args.input_header

        self._type_idx_dict = {"Src_V": 'src', "Dst_V": 'dst', "Edge": 'edge', "Null": "nullptr"}

        self._atomic = args.atomic

    @property  
    def output_name(self):
        if(self._output_name != ""):
            return os.path.join(self._output_dir, self._output_name)
        else:
            str = self._edge_op + "_" + self._gather_op + "_" + self._a_t + "_" + self._b_t + "_" + self._c_t + ".cu"
            return os.path.join(self._output_dir, str)

    @property 
    def file_header_name(self):
        return os.path.join(self._input_dir, self._input_header)

    # @property 
    # def tensor_idx(self, type):
    #     return  self._type_idx_dict[type]

    def gen_edge_tmp(self):
        a_idx = self._type_idx_dict[self._a_t]
        b_idx = self._type_idx_dict[self._b_t]
        # c_idx = self.tensor_idx(self._c_t)
        with open(self.output_name, "a") as f:
            str = "\tfloat edge_tmp = "
            if(self._edge_op == "add"):
                # str += "A[" + a_idx + "*Feat_Size + feat] + B[" + b_idx + "*Feat_Size + feat];"
                str += "A[" + a_idx + "*Feat_Size + feat] + B[" + b_idx + "*Feat_Size + feat];"
            elif (self._edge_op == "sub"):
                str += "A[" + a_idx + "*Feat_Size + feat] - B[" + b_idx + "*Feat_Size + feat];"
            elif (self._edge_op == "mul"):
                str += "A[" + a_idx + "*Feat_Size + feat] * B[" + b_idx + "*Feat_Size + feat];"
            elif (self._edge_op == "div"):
                str += "A[" + a_idx + "*Feat_Size + feat] / B[" + b_idx + "*Feat_Size + feat];"
            elif (self._edge_op == "copy_lhs"):
                str += "A[" + a_idx + "*Feat_Size + feat];"
            else:
                assert(self._edge_op == "copy_rhs")
                str += "B[" + b_idx + "*Feat_Size + feat];"

            str += '\n'
            f.write(str)

    def gen_C(self):
        c_idx = self._type_idx_dict[self._c_t]
        with open(self.output_name, "a") as f:
            if(self._atomic):
                str = "\t"                
            else:
                str = "\tC[" + c_idx + "*Feat_Size + feat] = "

            if(self._gather_op == "copy_lhs"):
                str += "C[" + c_idx + "*Feat_Size + feat];"
            elif (self._gather_op == "copy_rhs"):
                str += "edge_tmp;"
            elif (self._gather_op == "sum"):
                if(self._atomic):
                    str += "gpu_runtime::writeAdd(&C[" + c_idx + "*Feat_Size + feat], &edge_tmp);"
                else:
                   str += "C[" + c_idx + "*Feat_Size + feat] + edge_tmp;"
            elif (self._gather_op == "max"):
                if(self._atomic):
                    str += "MyatomicMax(&C[" + c_idx + "*Feat_Size + feat], &edge_tmp);"
                else:
                    str += "max(C[" + c_idx + "*Feat_Size + feat], edge_tmp);"
            elif (self._gather_op == "min"):
                if(self._atomic):
                    str += "MyatomicMin(&C[" + c_idx + "*Feat_Size + feat], &edge_tmp);"
                else:
                    str += "min(C[" + c_idx + "*Feat_Size + feat], edge_tmp);"
            elif (self._gather_op == "mean"):
                if(self._atomic):
                    str += "MyatomicAdd(&C[" + c_idx + "*Feat_Size + feat], &(edge_tmp/graph.d_get_degree(dst)));"
                else:
                    str += "C[" + c_idx + "*Feat_Size + feat] + edge_tmp/graph.d_get_degree(dst);"

            str += '\n'
            f.write(str)


    def generate_file(self):
        # if file exist
        if(os.path.exists(self.output_name)):
            os.remove(self.output_name)
        os.system("cp " + self.file_header_name + " " + self.output_name)
        # with open(self.output_name, "a") as f:
            
        self.gen_edge_tmp()
        self.gen_C()
        with open(self.output_name, "a") as f:
            f.write("}")

    def main(self):
        self.generate_file()

        # pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_op", default="copy_lhs", choices=['copy_lhs', 'copy_rhs', 'add', 'sub', 'mul', 'div'], help="edge op name", type=str)
    parser.add_argument("--gather_op", default="sum", choices=['copy_lhs', 'copy_rhs', 'sum', 'max', 'min', 'mean'], help="gather op name", type=str)
    parser.add_argument("--A_T", default="Src_V", choices=['Src_V', 'Dst_V', 'Edge'], help="Tensor A Type", type=str)
    parser.add_argument("--B_T", default="Src_V", choices=['Src_V', 'Dst_V', 'Edge'], help="Tensor B Type", type=str)
    parser.add_argument("--C_T", default="Dst_V", choices=['Src_V', 'Dst_V', 'Edge'], help="Tensor C Type", type=str)

    parser.add_argument("--output_dir", default="./output", help="Output Dir", type=str)
    parser.add_argument("--output_file_name", default="", help="Output File Name", type=str)

    parser.add_argument("--input_dir", default="./src", help="Input Dir", type=str)
    parser.add_argument("--input_header", default="header.cu", help="Input Header File Name", type=str)

    parser.add_argument("--atomic", default=1, help="Atomic Config", type=int)


    args = parser.parse_args()

    generator = Generator(args)
    generator.main()