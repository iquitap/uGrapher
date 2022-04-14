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
# from util import GLOBAL
import subprocess
import time
import re

time = time.strftime("%Y%m%d%H%M%S", time.localtime()) 
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(CURRENT_DIR, 'logs', time)
if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR)
import sys
sys.path.append(os.path.dirname(CURRENT_DIR))

from utility.globals import GLOBAL
class GNNTester:
    ''' Automation test entry '''

    def __init__(self, dataDir, dim, classes, gpu, training, 
    torch_profile, model, n_epochs, hidden, aggregator_type, 
    num_layers, num_heads, num_outheads,in_drop, attn_drop, negative_slope, residual, sparseTensor):
        self._dataDir = dataDir
        self._dim = dim
        self._classes = classes
        self._gpu = gpu
        self._training = training
        self._torch_profile = torch_profile
        self._model = model
        self._n_epochs = n_epochs
        self._hidden = hidden
        self._aggregator_type = aggregator_type
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._num_outheads = num_outheads
        self._in_drop = in_drop
        self._attn_drop = attn_drop
        self._negative_slope = negative_slope
        self._residual = residual 
        self._sparseTensor = sparseTensor

    @property
    def gat_hidden(self):
        return [8]
    
    @property
    def gin_hidden(self):
        return [64]
    
    @property
    def gcn_hidden(self):
        return [16]
    
    @property
    def sage_hidden(self):
        return [16]

    @property
    def dgl_pyg_dataset(self):
        dataset = [
                ('cora' 	        	, 1433	    , 7   ),  
                ('citeseer'	        , 3703	    , 6   ),  
                ('pubmed'	        	, 500	    , 3   ),      
                ('ppi'	            , 50	    , 121 ),   
        
                ( 'amazon0505'               , 96	  , 22),
                ( 'artist'                   , 100    , 12),
                ( 'com-amazon'               , 96	  , 22),
                ( 'soc-BlogCatalog'	         , 128    , 39), 
                ( 'amazon0601'  	         , 96	  , 22), 

                ('PROTEINS_full'             , 29       , 2) ,   
                ('OVCAR-8H'                  , 66       , 2) , 
                ('Yeast'                     , 74       , 2) ,
                ('DD'                        , 89       , 2) ,
                ('TWITTER-Real-Graph-Partial', 1323     , 2) ,   
                ('SW-620H'                   , 66       , 2) 
        ]
        return dataset

    @property
    def gat_head(self):
        return [8]

    @property
    def dgl_path(self):
        return os.path.join(CURRENT_DIR,'dgl_baseline','dglTester.py')


    @staticmethod
    def log2csv(log, framework):
        dataset_li = []
        time_li = []
        fp = open(log, "r")
        for line in fp:
            if "dataset=" in line:
                data = re.findall(r'dataset=.*?,', line)[0].split('=')[1].replace(",", "").replace('\'', "")
                # print(data)
                dataset_li.append(data)
            pattern = 'Time: (ms)' if framework == 'dgl' else '(ms):'
            if pattern in line:
                time = line.split(pattern)[1].rstrip("\n")
                # print(time)
                time_li.append(time)
        fp.close()

        fout = open(log.strip(".log") + ".csv", 'w')
        fout.write("dataset,Avg.Epoch (ms)\n")
        for data, time in zip(dataset_li, time_li):
            fout.write("{},{}\n".format(data, time))
        fout.close()
    
    @staticmethod
    def aggregator_transformer(framework, aggregator_type):
        '''
        There are equivalent types between dgl and pyg, we should do 
             Dgl      Pyg
             gcn  <-> add
             pool <-> max
             mean <-> mean
        ''' 
        if framework == 'dgl':
            if aggregator_type == 'add':
                return 'gcn'
            elif aggregator_type == 'max':
                return 'pool'
            else:
                return aggregator_type
        else:
            if aggregator_type == 'gcn':
                return 'add'
            elif aggregator_type == 'pool':
                return 'max'
            else:
                return aggregator_type            




    def dgl_baseline_test(self):
 
        ''' Entry for DGL test '''
        # def gcn_gin():


        for model in self._model:
            hidden = eval(f'self.{model}_hidden')
            for hid in hidden:
                for data, d, c in self.dgl_pyg_dataset:
                    for aggregator_type in self._aggregator_type:
                        aggregator_type = GNNTester.aggregator_transformer('dgl',aggregator_type)
                        command = f"python {self.dgl_path} --dataDir {self._dataDir} --dataset {data} --dim {d} --hidden {hid} --classes {c} --model {model} --training {self._training} --torch_profile {self._torch_profile} --gpu {self._gpu} --num_heads {self._num_heads} --aggregator_type {aggregator_type} --n_epochs {self._n_epochs}"
                        GLOBAL.printd(command)
                        log = os.path.join(LOG_DIR,f'dgl_{model}_{hid}_{self._training}.log')
                        with open(log,'a') as fp:
                            ret = subprocess.call(command, shell=True, stdout=fp)
                        GLOBAL.printd(ret)
                GNNTester.log2csv(log,'dgl')

    def pyg_baseline_test(self):
        pyg_gcn_gin_dir = os.path.join(CURRENT_DIR,'pyg_baseline','gcn_gin')
        pyg_gcn_path = os.path.join(pyg_gcn_gin_dir,'0_bench_pyg_gcn.py')
        pyg_gin_path = os.path.join(pyg_gcn_gin_dir,'0_bench_pyg_gin.py')
        pyg_gat_dir = os.path.join(CURRENT_DIR,'pyg_baseline','gat')
        pyg_gat_path = os.path.join(pyg_gat_dir,'0_bench_pyg_gat.py')
        # pyg_sage_dir = os.path.join(CURRENT_DIR,'pyg_baseline','sage')
        # pyg_sage_path = os.path.join(pyg_sage_dir,'0_bench_pyg_sage.py')

        profiling = False 
        if(profiling == True):
            py_file = "pyg_main_profiling.py"
        else:
            py_file = "pyg_main.py"

        for model in self._model:
            if model == 'gcn' or model == 'gin':
                os.chdir(pyg_gcn_gin_dir)
                hidden = eval(f'self.{model}_hidden')
            elif model == 'gat':
                os.chdir(pyg_gat_dir)
                hidden = eval(f'self.{model}_hidden')
            elif model == 'sage':
                # os.chdir(pyg_sage_dir)
                hidden = eval(f'self.{model}_hidden')
            for hid in hidden:
                for data, d, c in self.dgl_pyg_dataset:
                    if model == 'sage':
                        for aggregator_type in self._aggregator_type:
                            aggregator_type = GNNTester.aggregator_transformer('pyg',aggregator_type)
                            pyg_sage_dir = os.path.join(CURRENT_DIR,'pyg_baseline',f'sage_{aggregator_type}')
                            os.chdir(pyg_sage_dir)
                            command = "PYTORCH_JIT=0 python {} --dataDir {} --dataset {} \
                                    --dim {} --hidden {} --classes {} --aggregator_type {}\
                                    --model {} --training {} --sparseTensor {} --epochs {}".format(py_file, self._dataDir, data, d, hid, c, aggregator_type, model, self._training, self._sparseTensor, self._n_epochs)
                            GLOBAL.printd(command)
                            log = os.path.join(LOG_DIR, f'pyg_{model}_{hid}_{self._training}_{aggregator_type}.log')
                            with open(log, 'a') as fp:
                                ret = subprocess.call(command, shell=True, stdout=fp)
                            GLOBAL.printd(ret)
                            GNNTester.log2csv(log, 'pyg')
                    else:
                        if model == 'gcn' or model == 'gin':
                            command = "PYTORCH_JIT=0 python {} --dataDir {} --dataset {} \
                                        --dim {} --hidden {} --classes {}\
                                        --model {} --training {} --sparseTensor {} --epochs {}".format(py_file,
                                                                                                       self._dataDir,
                                                                                                       data, d, hid, c,
                                                                                                       model,
                                                                                                       self._training,
                                                                                                       self._sparseTensor,
                                                                                                       self._n_epochs)

                        elif model == 'gat':
                            command = "PYTORCH_JIT=0 python {} --dataDir {} --dataset {} \
                                    --dim {} --hidden {} --classes {}\
                                    --model {} --training {} --sparseTensor {} --num-heads {} --epochs {}".format(
                                py_file, self._dataDir, data, d, hid, c, model, self._training, self._sparseTensor,
                                self._num_heads, self._n_epochs)


                        GLOBAL.printd(command)
                        log = os.path.join(LOG_DIR, f'pyg_{model}_{hid}_{self._training}.log')
                        with open(log,'a') as fp:
                            ret = subprocess.call(command, shell=True, stdout=fp)
                        GLOBAL.printd(ret)
                        GNNTester.log2csv(log,'pyg')

        os.chdir(CURRENT_DIR)
        
        

    def load_balance_test(self):
        ''' Entry for load balance test '''

        workable_algo = ['gat','gat_8head','gat2_8head',
                              'sage_pooling']
        load_balance_test_dir = os.path.join(CURRENT_DIR, 'load_balance_test')
        os.chdir(load_balance_test_dir)
        for algo in workable_algo:

            GLOBAL.printh(f'GNNTester - Collecting {algo}')

            ret = subprocess.call(f'python nvprof.py --algo {algo} --device 0',shell=True)
            GLOBAL.printi(ret)

            ret = subprocess.call(f'python optimal.py --algo {algo} --device 0',shell=True)
            #GLOBAL.printi(ret)

            GLOBAL.printh(f'GNNTester - Finishes {algo}')
        os.chdir(CURRENT_DIR)

def str2bool(s):
    if isinstance(s, bool):
        return s
    elif s.lower() in ('true','t','yes','1'):
        return True
    elif s.lower() in ('false','f','n','0'):
        return False
    else:
        GLOBAL.printe('Unexpected arg value for torch_profile')
        raise argparse.ArgumentTypeError('Unexpected arg value for torch_profile')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataDir", type=str, default="/home/yjzhou/github/gnn/dataset/npz", help="the path to graphs")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--dim", type=int, default=96, help="input embedding dimension")
    parser.add_argument("--hidden", type=int, default=16, help="number of hidden feature size")
    parser.add_argument("--classes", type=int, default=10, help="number of output classes")
    parser.add_argument("--model", nargs='+', default=['gin'], choices=['gcn', 'gin', 'gat', 'sage'], help="type of model")
    parser.add_argument("--training", type=str, default='inference', choices=['training', 'inference'], help="training or inference")
    parser.add_argument("--aggregator_type", nargs='+', default=['mean'], choices=['mean', 'gcn', 'pool','add','max'], help="aggregator_type")
    parser.add_argument("--num_layers", type=int, default=1,
                            help="number of gat layers")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num_outheads", type=int, default=1,
                        help="number of output attention heads")                        
    parser.add_argument("--torch_profile", type=str2bool, default=False,  help="torch_profile")
    parser.add_argument("--in_drop", type=float, default=.0,
                            help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.0,
                            help="attention dropout")

    parser.add_argument('--negative-slope', type=float, default=0.2,
                            help="the negative slope of leaky relu")

    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument('--early_stop', action='store_true', default=False,
                            help="indicates whether to use early stop or not")
    parser.add_argument("--sparseTensor", type=int, default=1,
                            help="sparseTensor")

    parser.add_argument("--baseline", type=str, default='dgl', choices=['dgl', 'pyg', 'all'], help="baseline")
    args = parser.parse_args()

    # args = parser.parse_args()
    # GLOBAL.printd(args)
    gnn_tester = GNNTester(dataDir=args.dataDir,
                    model=args.model,
                    # dataset=args.dataset,
                    dim=args.dim,
                    classes=args.classes,
                    gpu=args.gpu,
                    training=args.training,
                    torch_profile=args.torch_profile,
                    n_epochs=args.n_epochs,
                    hidden=args.hidden,
                    aggregator_type=args.aggregator_type,
                    num_layers=args.num_layers,
                    num_heads=args.num_heads,
                    num_outheads=args.num_outheads,
                    in_drop=args.in_drop,
                    attn_drop=args.attn_drop,
                    negative_slope=args.negative_slope,
                    residual=args.residual,
                    sparseTensor=args.sparseTensor
                    )
   
    if(args.baseline == 'dgl'):
        gnn_tester.dgl_baseline_test()
    elif(args.baseline == 'pyg'):
        gnn_tester.pyg_baseline_test()
    else:
        gnn_tester.pyg_baseline_test()
        gnn_tester.dgl_baseline_test()
    
            

