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

#!/usr/bin/env python                                                           
#
#                                                                               

import opentuner
from opentuner import ConfigurationManipulator
from opentuner import EnumParameter
from opentuner import IntegerParameter
from opentuner import MeasurementInterface
from opentuner import Result
from sys import exit
import os
import argparse
import pandas as pd
import csv
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from util import GLOBAL

nvcc_bin = GLOBAL.nvcc_bin
gcpp_bin = GLOBAL.gcpp_bin
lib_dir = GLOBAL.lib_dir

fhid_mp = GLOBAL.fhid_mp

class UGrapherTuner(MeasurementInterface):
    num_sched = 0
    input_f = 0
    hidden_f = 0
    output_f = 0
    device = 0
    output_dir = ""
    algo = ""
    dataset = ""
    dataset_path = ""

    def manipulator(self):
        """                                                                          
        Define the search space by creating a                                        
        ConfigurationManipulator                                                     
        """

        algo_info = pd.read_csv("algo_info.csv", index_col="Algorithm")
        self.num_sched = algo_info.loc[self.args.algo]["num_schedule"]

        self.rec = []

        self.output_dir = self.args.output_dir
        self.algo = self.args.algo
        self.dataset = self.args.dataset

        self.dataset_path = GLOBAL.dataset_path(self.dataset)

        info = pd.read_csv("datainfo.csv", index_col="Dataset")
        self.input_f = info.loc[self.dataset]["input_f"]
        self.hidden_f = fhid_mp[self.algo]
        self.output_f = info.loc[self.dataset]["output_f"]

        manipulator = ConfigurationManipulator()

        for i in range(self.num_sched):
            manipulator.add_parameter(
                EnumParameter('ef_schedule_' + str(i), GLOBAL.schedules)
            )
            manipulator.add_parameter(IntegerParameter('group_size_' + str(i), 0, 6))
            manipulator.add_parameter(IntegerParameter('tile_size_' + str(i), 0, 6))

        return manipulator

    def compile(self, cfg,  id):
        """                                                                          
        Compile a given configuration in parallel                                    
        """

        cuda_output_file = self.output_dir + self.algo + ".gt" + ".cu"
        bin_output_file = cuda_output_file + ".o"

        if self.args.gpu_type == "V100":
            nvcc_command = nvcc_bin + " -ccbin " + gcpp_bin + "   -lcublas -rdc=true -DCTA_SIZE=512 -gencode arch=compute_70,code=sm_70 -std=c++11 -O3 -I " + lib_dir + "  -Xcompiler \"-w\" -Wno-deprecated-gpu-targets --use_fast_math -Xptxas \" -dlcm=ca --maxrregcount=64\" " + cuda_output_file + " -o " + bin_output_file
        elif args.gpu_type == "A100":
            nvcc_command = nvcc_bin + " -ccbin " + gcpp_bin + "   -lcublas -rdc=true -DCTA_SIZE=512 -gencode arch=compute_80,code=sm_80 -std=c++11 -O3 -I " + lib_dir + "  -Xcompiler \"-w\" -Wno-deprecated-gpu-targets --use_fast_math -Xptxas \" -dlcm=ca --maxrregcount=64\" " + cuda_output_file + " -o " + bin_output_file
        else:
            print(f"GPU type {self.args.gpu_type} is not supported!")
            exit()
        
        print(nvcc_command)

        return self.call_program(nvcc_command)

    def parse_running_time(self, log_file_name='test.out'):
        """Returns the elapsed time only, from the output file"""

        min_time = 10000

        with open(log_file_name) as f:
            content = f.readlines()
        content = [x.strip() for x in content]

        for line in content:
            time_str = line.strip()
            time = float(time_str)
            if time < min_time:
                min_time = time

        return min_time

    def run_precompiled(self, desired_result, input, limit, compile_result, id):
        """                                                                          
        Run a compile_result from compile() sequentially and return performance      
        """

        cfg = desired_result.configuration.data
        
        if compile_result['returncode'] != 0:
            print (str(compile_result))

        assert compile_result['returncode'] == 0

        log_file_name = 'test.out'
        # if not os.path.exists(log_file_name):
        f = open(log_file_name, 'w')
        f.close()

        device_command = "CUDA_VISIBLE_DEVICES=" + str(self.args.device) + " "
        bin_output_file = self.output_dir + self.algo + ".gt.cu.o "

        run_cmd = device_command + bin_output_file \
                        + self.dataset_path \
                        + str(self.input_f) + " " \
                        + str(self.hidden_f) + " " \
                        + str(self.output_f) + " 1 1"

        for i in range(self.num_sched):
            run_cmd = run_cmd + " 1 " + cfg['ef_schedule_' + str(i)]

            groupsz = cfg['group_size_' + str(i)]
            groupsz = 1 << groupsz
            run_cmd = run_cmd + " " + str(groupsz)

            tilesz = cfg['tile_size_' + str(i)]
            tilesz = 1 << tilesz
            run_cmd = run_cmd + " " + str(tilesz)

        run_cmd = run_cmd + " ./test.out"

        print ("run_cmd: " + run_cmd)

        # default value -1 for memory_limit translates into None (no memory upper limit)
        # setting memory limit does not quite work yet
        process_memory_limit = None
        if self.args.memory_limit != -1:
            process_memory_limit = self.args.memory_limit
        # print ("memory limit: " + str(process_memory_limit))
        run_result = self.call_program(run_cmd, limit=self.args.runtime_limit, memory_limit=process_memory_limit)

        if run_result['timeout'] == True:
            val = self.args.runtime_limit
        else:
            val = self.parse_running_time();
        
        self.call_program('rm test.out')
        print ("run result: " + str(run_result))
        print ("running time: " + str(val))

        if run_result['timeout'] == True:
            print ("Timed out after " + str(self.args.runtime_limit) + " seconds")
            return opentuner.resultsdb.models.Result(time=val)
        elif run_result['returncode'] != 0:
            if self.args.killed_process_report_runtime_limit == 1 and run_result['stderr'] == 'Killed\n':
                print ("process killed " + str(run_result))
                return opentuner.resultsdb.models.Result(time=self.args.runtime_limit)
            else:
                print (str(run_result))
                # exit()
                return opentuner.resultsdb.models.Result(time=val)
        else:
            self.rec.append(val)
            return opentuner.resultsdb.models.Result(time=val)

    def compile_and_run(self, desired_result, input, limit):
        """                                                                          
        Compile and run a given configuration then                                   
        return performance                                                           
        """

        cfg = desired_result.configuration.data

        # this pases in the id 0 for the configuration
        compile_result = self.compile(cfg, 0)
        # print "compile_result: " + str(compile_result)
        return self.run_precompiled(desired_result, input, limit, compile_result, 0)

    def save_final_config(self, configuration):
        """called at the end of tuning"""
        print ('Final Configuration:', configuration.data)
        fig = plt.figure()
        ax = fig.add_subplot()
        timeline = np.array(list(range(len(self.rec))))
        ax.plot(timeline, self.rec)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (s)')
        name = self.algo + "_tune_result.png"
        fig.savefig(name)
        self.manipulator().save_to_file(configuration.data, self.algo + '_final_config.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=opentuner.argparsers())
    parser.add_argument("--algo", default="gat", help="algorithm name", type=str)
    parser.add_argument("--dataset", default='cora', help="dataset name", type=str)
    parser.add_argument("--output_dir", default="./end2end/src/", help="output dir", type=str)
    parser.add_argument("--gpu_type", default="V100", help="gpu type", type=str)
    parser.add_argument("--device", default=0, help="gpu device", type=int)
    parser.add_argument('--runtime_limit', type=float, default=300, help='a limit on the running time of each program')
    parser.add_argument('--memory_limit', type=int, default=-1,help='set memory limit on unix based systems [does not quite work yet]')    
    parser.add_argument('--killed_process_report_runtime_limit', type=int, default=0, help='reports runtime_limit when a process is killed by the shell. 0 for disable (default), 1 for enable')
    args = parser.parse_args()
    # pass the argumetns into the tuner
    UGrapherTuner.main(args)
    