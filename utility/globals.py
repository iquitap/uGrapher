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

# from GNNCompiler.utility.myLogger import *
from utility.myLogger import *

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
        self.gcpp_bin = "/usr/bin/g++"

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
