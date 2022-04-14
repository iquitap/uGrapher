# uGrapher

This repository contains the source code for a research paper.


## Environment Preparation.

To evaluate uGrapher, we use two different GPUs as our hardware platforms: Tesla V100 and Ampere A100.
The details of the environment are as follows:

**NVIDIA GPU V100**
+ Hardware Requirements
    1. CPU: Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz
    2. NVIDIA Tesla V100
+ Software Requirements
    1. Ubuntu 18.04.5 (kernel 5.4.0)
    2. GPU Driver: 460.39
    3. CUDA 11.1
    4. Anaconda3-2020.7
    5. Pytorch 1.8.0

**NVIDIA GPU A100**
+ Hardware Requirements
    1. CPU: Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz
    2. NVIDIA Ampere A100
+ Software Requirements
    1. Ubuntu 20.04.1 (Kernel 5.8.0)
    2. GPU Driver: 460.39
    3. CUDA 11.1
    4. Anaconda3-2020.7
    5. Pytorch 1.8.0


## Getting Started Instructions.

### **Step-1: Clone this project**
```
git clone https://github.com/iquitap/uGrapher.git
```

### **Step-2: Environment Setup**
There are two ways to setup the environment of uGrapher and baselines.
#### Method 1:  Setup the environment via Docker.
+ Install Docker Engine with NVIDIA GPU Support **[Toturial](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)**.
    ```shell
    curl https://get.docker.com | sh \
    && sudo systemctl --now enable docker

    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker
    ```
+ `cd Docker` then either goto `V100/` or `A100/`.
+ Run `./build.sh`, it may takes a while (around 20 minutes) for building the container.
+ Run `./launch.sh` then it will bring up an new interactive command line interface.

#### Method 2:  Setup via conda and pip.

1. Install Anaconda3 as the Python runtime
    ```shell
    $ cd ~ && mkdir -p .local && cd .local
    $ wget -O Anaconda3-2020.07-Linux-x86_64.sh https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
    $ chmod +x Anaconda3-2020.07-Linux-x86_64.sh
    $ ./Anaconda3-2020.07-Linux-x86_64.sh -b -p ../.local/anaconda3
    ```

2. Create a **`conda`** environment.
    ```shell
    $ eval "$($HOME/.local/anaconda3/bin/conda shell.zsh hook)"
    $ # cd into uGrapher repository
    $ conda create --name uGrapher python=3.8
    $ conda activate uGrapher
    ```
3. Install [**`Pytorch`**](https://pytorch.org/get-started/locally/).
    ```shell
    $ conda install pytorch=1.8 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
    ```
4. Install [**`Deep Graph Library (DGL)`**](https://github.com/dmlc/dgl).
    ```shell
    $ conda install -c dglteam dgl-cuda11.1=0.6
    $ pip install --upgrade torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
5. Install [**`Pytorch-Geometric (PyG)`**](https://github.com/rusty1s/pytorch_geometric).
    ```shell
    $ pip install --upgrade torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
    $ pip install --upgrade torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
    $ pip install --upgrade torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
    $ pip install --upgrade torch-geometric

    ```

### **Step-3: Download the graph datasets.**
+ Graph datasets can be downloaded via this **[[LINK](https://drive.google.com/file/d/1_FHpKAg6CTY4EE-t04TgBZA8c9Umpw1K/view?usp=sharing)]** (filename: `gnn_dataset.tar.gz`).
+ Unzip the graph datasets `tar -zxvf gnn_dataset.tar.gz` at the project root directory.


## Evaluation Instructions.

### **Running Baseline** 

+ We provide a uniform test script `GNNTester.py` for baseline evaluation.     
+ For example, the command `python GNNTester.py --model GCN --baseline dgl --hidden 16`  can be used to evaluate the execution time of GCN on DGL with hidden feature of 16.  
+ The command `python GNNTester.py -h` can be used to view the help information.


### **Running uGrapher**

#### **End-to-end evaluation**
We provide a test script `ugrapher/run_optimal.py` for end-to-end evaluation.   

This script would compile and run the CUDA code *<#Algorithm name>.gt.cu* in "./ugrapher/end2end/src/".

The script utilizes the results in "./ugrapher/benchmark_opt_sched/<#GPU type>/all_optimal_sched.csv" to decide the schedules it would run.

Parameters:

- "--algo" [type=str]: <#Algorithm name>, now supporting [gat, gcn_v1, gcn_v2, gin, sage_mean_v1, sage_mean_v2, sage_pooling, sage_sum_v1, sage_sum_v2]. 
- "--gpu_type" [type=str] (optional): <#GPU type> parameter, specifying which device the data to get optimal choices of edge function schedules come from, now supporting "V100" (default) and "A100".
- "--dataset" [type=str] (optional): <#Dataset name>, now supporting [cora, citeseer, pubmed, ppi, amazon0505, artist, com-amazon, soc-BlogCatalog, amazon0601, PROTEINS_full, OVCAR-8H, Yeast, DD, TWITTER-Real-Graph-Partial, SW-620H]. Leave blank to run all.
- "--device" [type=int] (optional): GPU device your code running on. Please check "nvidia-smi" in the terminal before launching *run_optimal.py* to assure that no other codes are running on the GPU core you've specified by this parameter.

#### **Case analysis**
We provide a test script `ugrapher/grid_search.py` for case study and profiling.   

This script would compile the CUDA code *<#Algorithm name>\_sched\_<#index>.gt.cu* in "./ugrapher/case_analysis/src/", and profile runtime information by *nvprof*.

Generated profiling files are in "./ugrapher/nvprof/", under the sub file folder "<#Edge function schedule name>/<#Algorithm name>\_sched\_<#index>/", named by *<#Algorithm name>\_sched\_<#index>\_<#Dataset name>\_h<#Hidden feature size>\_groupsz<#Group size>\_tilesz<#Tiling size>\_<#pid>.csv*.

Generated runtime information files are in "./ugrapher/case_analysis/result/<#Algorithm name>_ef_prof.csv", which records the mean runtime of the edge function schedule.

Parameters:

- "--algo" [type=str]: <#Algorithm name>, now supporting [gat, gcn_v1, gcn_v2, gin, sage_mean_v1, sage_mean_v2, sage_pooling, sage_sum_v1, sage_sum_v2].
- "--gpu_type" [type=str] (optional): <#GPU type> parameter, specifying which device the data to get optimal choices of edge function schedules come from, now supporting "V100" (default) and "A100".
- "--enable_nvprof" [type=int] (optional): whether to use *nvprof*. 0 (default) for disable, 1 for enable.
- "--dataset" [type=str] (optional): <#Dataset name>, now supporting [cora, citeseer, pubmed, ppi, amazon0505, artist, com-amazon, soc-BlogCatalog, amazon0601, PROTEINS_full, OVCAR-8H, Yeast, DD, TWITTER-Real-Graph-Partial, SW-620H]. Leave blank to run all.
- "--device" [type=int] (optional): GPU device your code runs on. Please check "nvidia-smi" in the terminal before launching *grid_search.py* to assure that no other codes are running on the GPU core you've specified by this parameter.
- "--test_ef" [type=int] (optional): specify how to record edge function runtime. 1 (default) for default testing, 2 for debugging.
- "--schedule" [type=str] (optional): specify a particular edge function schedule to profile, now supporting [t_vertex_group_tiling, w_vertex_group_tiling, t_edge_group_tiling, w_edge_group_tiling]. Leave blank to test all.
- "--group_size" [type=int] (optional): <#Group size> parameter for the edge function schedules. Leave blank to test over all predefined group sizes, i.e. [1, 2, 4, 8, 16, 32, 64]. Other group sizes are welcome, but are recommended to be power of 2.
- "--par_tiling" [type=int] (optional): <#Tiling size> parameter for the edge function schedules. Leave blank to test over all predefined tiling sizes, i.e. [1, 2, 4, 8, 16, 32, 64]. Other tiling sizes are welcome, but are recommended to be power of 2.

#### **AutoTuner**
We provide a test script `ugrapher/autotune.py` for automatical optimal schedule searching.   

This script would compile and run the CUDA code *<#Algorithm name>.gt.cu* in "./ugrapher/end2end/src/".

The script automatically searches for the optimal edge function schedules for a particular algorithm on a particular dataset.

Generated optimal edge function schedules found by this script are in "./ugrapher/<#Algorithm name>_final_config.json", and searching history is plotted in "./ugrapher/<#Algorithm name>_tune_result.png".

Parameters:

- "--algo" [type=str]: <#Algorithm name>, now supporting [gat, gcn_v1, gcn_v2, gin, sage_mean_v1, sage_mean_v2, sage_pooling, sage_sum_v1, sage_sum_v2]. 
- "--gpu_type" [type=str] (optional): <#GPU type> parameter, specifying which device the data to get optimal choices of edge function schedules come from, now supporting "V100" (default) and "A100".
- "--dataset" [type=str] (optional): <#Dataset name>, now supporting [cora, citeseer, pubmed, ppi, amazon0505, artist, com-amazon, soc-BlogCatalog, amazon0601, PROTEINS_full, OVCAR-8H, Yeast, DD, TWITTER-Real-Graph-Partial, SW-620H].
- "--device" [type=int] (optional): GPU device your code running on. Please check "nvidia-smi" in the terminal before launching *run_optimal.py* to assure that no other codes are running on the GPU core you've specified by this parameter.
- "--stop-after" [type=int]: the total time cost (in seconds) upperbound for running this script.


## License

This project is licensed under the Apache-2.0 License.
