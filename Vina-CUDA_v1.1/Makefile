# Need to be modified according to different users
WORK_DIR=/your/path/to/Autodock-Vina-CUDA-master
BOOST_LIB_PATH=/your/path/to/boost_1_77_0
NVCC_COMPILER=/your/path/to/cuda-12.2/bin/nvcc
GRID_DIM=-DGRID_DIM1=64 -DGRID_DIM2=128
DOCKING_BOX_SIZE=-DLARGE_BOX

# Should not be modified
BOOST_INC_PATH=-I$(BOOST_LIB_PATH) -I$(BOOST_LIB_PATH)/boost 
VINA_GPU_INC_PATH=-I$(WORK_DIR)/lib -I$(WORK_DIR)/inc/ -I$(WORK_DIR)/inc/cuda
OPENCL_INC_PATH=
LIB1=-lboost_program_options -lboost_system -lboost_filesystem 
LIB2=-lstdc++ -lstdc++fs
LIB3=-lm -lpthread
LIB_PATH=-L$(BOOST_LIB_PATH)/stage/lib 
SRC=./lib/*.cpp $(BOOST_LIB_PATH)/libs/thread/src/pthread/thread.cpp $(BOOST_LIB_PATH)/libs/thread/src/pthread/once.cpp 
SRC_CUDA = ./inc/cuda/kernel1.cu ./inc/cuda/kernel2.cu 
MACRO=$(GRID_DIM) #-DDISPLAY_SUCCESS -DDISPLAY_ADDITION_INFO

all:out
out:./main/main.cpp
	$(NVCC_COMPILER) -o Vina-GPU-2-1-CUDA $(BOOST_INC_PATH) $(VINA_GPU_INC_PATH) $(OPENCL_INC_PATH) $(DOCKING_BOX_SIZE) ./main/main.cpp -O3 $(SRC) $(SRC_CUDA) $(LIB1) $(LIB2) $(LIB3) $(LIB_PATH) $(OPTION) $(MACRO) -DNDEBUG 
source:./main/main.cpp
	$(NVCC_COMPILER) -o Vina-GPU-2-1-CUDA $(BOOST_INC_PATH) $(VINA_GPU_INC_PATH) $(OPENCL_INC_PATH) $(DOCKING_BOX_SIZE) ./main/main.cpp -O3 $(SRC) $(SRC_CUDA) $(LIB1) $(LIB2) $(LIB3) $(LIB_PATH) $(OPTION) $(MACRO) -DNDEBUG -DBUILD_KERNEL_FROM_SOURCE
clean:
	rm Vina-GPU-2-1-CUDA
