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
from util import GLOBAL

class Grid_Search:
    ''' Run nvprof to collect the inference results of different algorithm '''
    def __init__(self, enable_nvprof, gpu_type, algo, dataset, test_ef, schedule, groupsz, tilesz, device, output_dir):
        self._use_nvprof = enable_nvprof
        self._algo = algo
        self._dataset = dataset
        self._schedule = schedule
        self._groupsz = groupsz
        self._tilesz = tilesz
        self._device = device
        self._output_dir = output_dir
        self._testef = test_ef
        self._dataset_info = self.dataset_info
        self._gpu_type = gpu_type
        self.complex_schedule_set = GLOBAL.schedules
        
        self.fhid_mp = GLOBAL.fhid_mp

    @property
    def algoset(self):
        ''' Set algorithm list '''
        algoset = list()
        algo_info = pd.read_csv("algo_info.csv", index_col="Algorithm")
        if not self._algo:
            GLOBAL.printe('No algorithm specified')
            exit()

        num_sched = algo_info.loc[self._algo]["num_schedule"]
        for i in range(num_sched):
            algoset.append(f'{self._algo}_sched_{str(i)}')

        return algoset

    @property
    def dataset(self):
        ''' Set dataset list '''
        if self._dataset:
            dataset = self._dataset
        else:
            dataset = GLOBAL.datasets

        return dataset

    @property
    def scheduleset(self):
        ''' Set scheduleset list '''
        if self._schedule:
            scheduleset = self._schedule
            sched_not_supported = [x for x in scheduleset if x not in self.complex_schedule_set]
            if len(sched_not_supported) > 0:
                for x in sched_not_supported:
                    GLOBAL.printe(f'Schedule {x} is not supported')
                exit()
        else:
            scheduleset = self.complex_schedule_set

        return scheduleset

    @property
    def groupszset(self):
        ''' Set groupszset list'''
        groupszset = []
        if self._groupsz == 0:
            groupszset = [1, 2, 4, 8, 16, 32, 64]
        else:
            groupszset.append(self._groupsz)
        return groupszset

    @property
    def tileszset(self):
        ''' Set tilingszset list'''
        tileszset = []
        if self._tilesz == 0:
            tileszset = [1, 2, 4, 8, 16, 32, 64]
        else:
            tileszset.append(self._tilesz)
        return tileszset

    @property
    def dataset_info(self):
        info = pd.read_csv("datainfo.csv", index_col="Dataset")
        return info

    @property
    def device_command(self):
        device_command = f"CUDA_VISIBLE_DEVICES={str(self._device)} "
        return device_command

    @staticmethod
    def append_csv(csv_path, content):
        with open(csv_path, 'a') as csv_file:
            csv_file.write(content)

    def emit(self, run_bin_command, nvprof_output_file):
        if self._use_nvprof:
            nvprof_command = f'{GLOBAL.nvprof_bin} --profile-child-processes --csv --log-file {nvprof_output_file}_%p.csv '

            all_command = self.device_command + nvprof_command + run_bin_command
        else:
            all_command = run_bin_command
        
        GLOBAL.printi(all_command)
        os.system(all_command)

    def main(self):
        ''' main process'''

        src_dir = './case_analysis/src/'
        result_path = os.path.join('case_analysis','result')
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        f_hid = self.fhid_mp[self._algo]

        for algo in self.algoset:
            GLOBAL.printh(f'Running {algo}')
            
            ef_prof = os.path.join(result_path, f'{algo}_ef_prof.csv')

            # Create ef_prof header & create output dir
            if self._testef == 1:
                GLOBAL.printi('Create ef_prof header & create output dir')
                with open(ef_prof, 'w') as result_file:
                    result_file.write('"Algorithm","Dataset","f_hidden"')
                    schd_cnt = len(self.scheduleset)
                    for schedule in self.scheduleset:

                        if self._use_nvprof:
                            output_dir = os.path.join(self._output_dir, schedule, algo)
                            if os.path.exists(output_dir):
                                shutil.rmtree(output_dir)
                            GLOBAL.printi(f'makedir {output_dir}')
                            os.mkdir(output_dir)

                        if schedule in self.complex_schedule_set:
                            schd_cnt = schd_cnt + len(self.groupszset)*len(self.tileszset) - 1
                            for groupsz in self.groupszset:
                                for tilesz in self.tileszset:
                                    schedule_p = f'{schedule}(g={str(groupsz)},t={str(tilesz)})'
                                    result_file.write(f',"{schedule_p}"')
                        else:
                            result_file.write(f',"{schedule}"')

                    result_file.write(f'\n,,{",ms"*schd_cnt}\n"{algo}"')
                GLOBAL.printi(f'{ef_prof} is created')

            GLOBAL.printi('Building CUDA...')
            nvcc_cu_command = f"python nvcc_cu.py --algo {algo} --gpu_type {self._gpu_type} --output_dir {src_dir}\n"
            GLOBAL.printi(f'{nvcc_cu_command}')
            os.system(nvcc_cu_command)

            for data in self.dataset:
                f_in, f_out = self._dataset_info.loc[data]["input_f"], self._dataset_info.loc[data]["output_f"]
                GLOBAL.printi(f'data: {data}\tf_in:{f_in}\tf_out:{f_out}')

                if self._testef != 0:
                    Grid_Search.append_csv(ef_prof, ",\"" + data + "\"," + str(f_hid))
                    for schedule in self.scheduleset:
                        output_dir = os.path.join(self._output_dir, schedule, algo)
                        nvprof_output_file = os.path.join(output_dir, f'{algo}_{data}_h{f_hid}')
                        run_bin_command = "python run_bin.py --algo " + algo \
                                        + " --output_dir " + src_dir \
                                        + " --device " + str(args.device) \
                                        + " --dataset " + data \
                                        + " --input_f " + str(f_in) \
                                        + " --hidden_f " + str(f_hid) \
                                        + " --output_f " + str(f_out) \
                                        + " --run_vf " + str(0) \
                                        + " --schedule " + schedule \
                                        + " --runtime_path " + ef_prof

                        if schedule in self.complex_schedule_set:
                            for groupsz in self.groupszset:
                                for tilesz in self.tileszset:
                                    run_bin_command_p = run_bin_command + " --group_size " + str(groupsz) + " --par_tiling " + str(tilesz)
                                    nvprof_output_file_p = nvprof_output_file + "_groupsz" + str(groupsz) + "_tilesz" + str(tilesz)
                                    Grid_Search.append_csv(ef_prof, f',')
                                    self.emit(run_bin_command_p, nvprof_output_file_p)
                        else:
                            Grid_Search.append_csv(ef_prof, f',')
                            self.emit(run_bin_command, nvprof_output_file)

                    Grid_Search.append_csv(ef_prof, "\n")
            
            GLOBAL.printi(f'{algo} Finished...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default=None, help="algorithm name", type=str)
    parser.add_argument("--enable_nvprof", default=0, help="whether to use nvprof", type=int)
    parser.add_argument("--gpu_type", default="V100", help="gpu type", type=str)
    parser.add_argument("--dataset", nargs="+", default=None, help="dataset name", type=str)
    parser.add_argument("--device", default=0, help="gpu device", type=int)
    parser.add_argument("--output_dir", default="./nvprof/", help="nvprof output_dir", type=str)
    parser.add_argument("--test_ef", default=1, help="how to test edge functions, 0 for disable, 1 for full testing, 2 for debugging", type=int)
    parser.add_argument("--schedule", nargs="+", default=None, help="gnn load balance schedule", type=str)
    parser.add_argument("--group_size", default=0, help="group size parameter for edge group load balance", type=int)
    parser.add_argument("--par_tiling", default=0, help="parameter for tiling feature dimension", type=int)
    args = parser.parse_args()

    gridSearch = Grid_Search(enable_nvprof = args.enable_nvprof, gpu_type=args.gpu_type, algo=args.algo, dataset=args.dataset, test_ef=args.test_ef, 
        schedule=args.schedule, device=args.device, output_dir=args.output_dir, groupsz=args.group_size, tilesz=args.par_tiling)
    gridSearch.main()
