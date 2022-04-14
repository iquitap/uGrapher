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

import logging
import logging.handlers
import os


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_DIR = os.path.join(CURRENT_DIR, "logs")
if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR)
###################### configure the log format ###############################################
# Configure main log format
logger = logging.getLogger('GNNTester')
logger.setLevel(logging.DEBUG)

# A handler class which writes formatted logging records to disk files.
fh_info_path = os.path.join(LOG_DIR,"GNNTester.txt")
fh_err_path = os.path.join(LOG_DIR,"GNNTester.txt")
# maxBytes = 5MB
fh_info = logging.handlers.RotatingFileHandler(fh_info_path,maxBytes=5*1024*1024, backupCount=5)
fh_err = logging.handlers.RotatingFileHandler(fh_err_path,maxBytes=5*1024*1024, backupCount=5)

# create another handler, for stdout in terminal
# A handler class which writes logging records to a stream
sh = logging.StreamHandler()
sh.setLevel(logging.ERROR)

# set formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh_info.setFormatter(formatter)
fh_err.setFormatter(formatter)
sh.setFormatter(formatter)

# set filter
info_filter = logging.Filter()
info_filter.filter = lambda record: record.levelno <= logging.WARNING
err_filter = logging.Filter()
err_filter.filter = lambda record: record.levelno > logging.WARNING
fh_info.addFilter(info_filter)
fh_err.addFilter(err_filter)

# add handler to logger
logger.addHandler(fh_info)
logger.addHandler(fh_err)
logger.addHandler(sh)

class Bcolors:
    ''' Used to highlight the text output '''
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class _Globals:
    ''' Defines global variables '''
    def __init__(self):
        self.log_handler = logger
        self.nvprof_bin = '/usr/local/cuda-11.1/bin/nvprof'
        self.nvcc_bin = "/usr/local/cuda-11.1/bin/nvcc"
        self.gcpp_bin = "/usr/bin/g++-7"
        self.lib_dir = "./src/runtime_lib"

        self.fhid_mp = {"gat":8, "gin":64, "gcn_v1":16, 
            "gcn_v2":16, "sage_mean_v1":16, "sage_mean_v2":16, 
            "sage_pooling":16, "sage_sum_v1":16, "sage_sum_v2":16}

        self.schedules = ["t_vertex_group_tiling", "w_vertex_group_tiling", 
            "t_edge_group_tiling", "w_edge_group_tiling"]

        self.datasets = ["cora", "citeseer", "pubmed", "ppi", "amazon0505", "artist", "com-amazon", "soc-BlogCatalog",
                       "amazon0601", "PROTEINS_full", "OVCAR-8H", "Yeast", "DD", "TWITTER-Real-Graph-Partial",
                       "SW-620H"]

    def dataset_path(self, dataset):
        if dataset in ["cora", "citeseer", "pubmed"]:
            dataset_dir = "/home/yxsong/GLpIR/dataset/misc/"
        elif dataset in ["amazon0505", "artist", "com-amazon", "soc-BlogCatalog", "amazon0601"]:
            dataset_dir = "/home/yxsong/GLpIR/dataset/osdi-ae-graphs-mtx/"
        else:
            dataset_dir = "/home/yjzhou/github/gnn/compiler/G2_artifact_eval/gnn_dataset/gnn_misc/ae_"
        
        return dataset_dir + dataset + ".mtx "

    def print(self, txt='', color=None, level=logging.INFO):
        if color:
            print(f"{color}{txt}{Bcolors.ENDC}")
        else:
            print(f"{txt}")
        logger.log(level=level,msg=txt)

    def printe(self, txt=''):
        ''' print Errors '''
        self.print(txt, level=logging.ERROR,color=Bcolors.FAIL)

    def printi(self, txt=''):
        ''' print Info '''
        self.print(txt, level=logging.INFO, color=None)

    def printh(self, txt=''):
        ''' print Header '''
        txt = f'\n{"="*70}\n{txt}\n{"="*70}'
        self.print(txt, level=logging.INFO, color=Bcolors.OKBLUE)

    def printd(self, txt=''):
        ''' print DEBUG '''
        self.print(txt, level=logging.DEBUG, color=Bcolors.OKCYAN)



GLOBAL = _Globals()
