# Vina-CUDA
## Introduction
In this project, we propose **Vina-CUDA** based on [Vina-GPU 2.1](https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1), which aims to further enhance Vina-GPU 2.1's docking speed by optimizing its core algorithms through deep utilization of GPU hardware features, thereby reducing the cost of virtual screening and increasing the efficiency of drug design. 
![Vina-CUDA](./image/Vina-CUDA.jpg)

## The Acceleration and Accuracy of Vina-CUDA
* The runtime acceleration of Vina-CUDA comprae with the Vina-GPU 2.1, QuickVina 2-GPU 2.1 and QuickVina-W-GPU 2.1 in Autodock-GPU, CASF-2016, PPARG, Astex, and PoseBuster librarys.

* Accuracy comparison of Vina-CUDA on AutoDock-GPU and CASF-2016 library.

 ## Compiling and Running Methods
 ### Linux
 **Note**: At least 8M stack size is needed. To change the stack size, use `ulimit -s 8192`.
 1. install [boost library](https://www.boost.org/) (Current Version is 1.77.0)
 2. install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (Current Version: v12.2)   **Note**: CUDA library can be usually in `/usr/local/cuda` for NVIDIA GPU cards.
 3. Open the Makefile file and change the following information: 
       1. `$WORK_DIR` : Set as your working directory (eg: path/of/your/work/directory/Vina-CUDA);
       2. `$BOOST_LIB_PATH` : Set to the path where the BOOST library is located (eg: path/of/BOOST/boost_1_77_0);
       3. `$NVCC_COMPILER` : Set to the path of the NVCC compiler (eg: /usr/local/cuda-12.2/bin/nvcc).
 4. Save the Makefile and type `make clean` and `make source -jthread` to build `$(VINA_CUDA_METHODS)` that compile the kernel files on the fly (this would take some time at the first use).
 5. After a successful compiling (there may be some warnings about the BOOST library, which can usually be ignored without affecting the normal operation of the programme), you will see the `$(Vina-GPU-2-1-CUDA)` in the work directory.
 6. In the work directory,type `$(Vina-GPU-2-1-CUDA) --config ./input_file_example/1u1b_config.txt` to run the Vina-CUDA method.
 7. once you successfully run `$(VINA_CUDA_METHODS)`, its runtime can be further reduced by typing `make clean` and `make` to build it without compiling kernel files.
 8. other compile options:

|Options| Description|
|--|--|
| -g | debug|
|-DTIME_ANALYSIS|output runtime analysis in `gpu_runtime.log`|
|-DDISPLAY_ADDITION_INFO|print addition information|
|-GRID_DIM|set the grid size (the value of `GRID_DIM1*GRID_DIM2`,eg. 64*128 must equal to the value of `thread`(eg. 8192) parameter)|
|-DSAMLL_BOX|the volume of the searching box less than 30/30/30 (will take less GPU memory)|
|-DLARGE_BOX|the volume of the searching box less than 70/70/70 (will take more GPU memory)


## Usage
|Arguments| Description|Default value
|--|--|--|
|--config | the config file (in .txt format) that contains all the following arguments for the convenience of use| no default
| --receptor | the recrptor file (in .pdbqt format)| no default
|--ligand| the ligand file (in .pdbqt fotmat)| no default
|--ligand_directory| this path specifies the directory of all the input ligands(in .pdbqt format) | no default
|--output_directory| this path specifies the directory of the output ligands | no default
|--thread| the scale of parallelism (docking lanes)|8192
|--search_depth| the number of searching iterations in each docking lane| heuristically determined
|--center_x/y/z|the center of searching box in the receptor|no default
|--size_x/y/z|the volume of the searching box|no default 
