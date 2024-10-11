# Need to be modified according to different users
WORK_DIR=/mnt/disk4/DingXiaoyu/Autodock-Vina-CUDA
BOOST_LIB_PATH=/mnt/disk4/DingXiaoyu/Vina-GPU-2.1-old/boost_1_77_0
NVCC_COMPILER=/mnt/disk4/DingXiaoyu/Vina-GPU-2.1-CUDA/cuda-12.2/bin/nvcc
DOCKING_BOX_SIZE=-DLARGE_BOX


#/usr/local/cuda-11.5 /mnt/disk4/DingXiaoyu/Vina-GPU-2.1/cuda-12.2
# Should not be modified
BOOST_INC_PATH=-I$(BOOST_LIB_PATH) -I$(BOOST_LIB_PATH)/boost 
VINA_GPU_INC_PATH=-I$(WORK_DIR)/lib -I$(WORK_DIR)/inc/ -I$(WORK_DIR)/inc/cuda
OPENCL_INC_PATH=
LIB1=-lboost_program_options -lboost_system -lboost_filesystem 
LIB2=-lstdc++ -lstdc++fs
LIB3=-lm -lpthread
LIB_PATH=-L$(BOOST_LIB_PATH)/stage/lib 
SRC=./lib/*.cpp $(BOOST_LIB_PATH)/libs/thread/src/pthread/thread.cpp $(BOOST_LIB_PATH)/libs/thread/src/pthread/once.cpp #../boost_1_77_0/boost/filesystem/path.hpMACRO=-DAMD_PLATFORM -DDISPLAY_SUCCESS -DDISPLAY_ADDITION_INFO
SRC_CUDA = ./inc/cuda/kernel1.cu ./inc/cuda/kernel2.cu 
#MACRO=$(OPENCL_VERSION) $(GPU_PLATFORM) -DBOOST_TIMER_ENABLE_DEPRECATED #-DDISPLAY_SUCCESS -DDISPLAY_ADDITION_INFO
MACRO=
all:out
out:./main/main.cpp
	$(NVCC_COMPILER) -o Vina-GPU-2-1-CUDA $(BOOST_INC_PATH) $(VINA_GPU_INC_PATH) $(OPENCL_INC_PATH) $(DOCKING_BOX_SIZE) ./main/main.cpp -O3 $(SRC) $(SRC_CUDA) $(LIB1) $(LIB2) $(LIB3) $(LIB_PATH) $(MACRO) $(OPTION) -DNDEBUG 
source:./main/main.cpp
	$(NVCC_COMPILER) -o Vina-GPU-2-1-CUDA $(BOOST_INC_PATH) $(VINA_GPU_INC_PATH) $(OPENCL_INC_PATH) $(DOCKING_BOX_SIZE) ./main/main.cpp -O3 $(SRC) $(SRC_CUDA) $(LIB1) $(LIB2) $(LIB3) $(LIB_PATH) $(MACRO) $(OPTION) -DNDEBUG -DBUILD_KERNEL_FROM_SOURCE
debug:./main/main.cpp
	$(NVCC_COMPILER) -o Vina-GPU-2-1-CUDA $(BOOST_INC_PATH) $(VINA_GPU_INC_PATH) $(OPENCL_INC_PATH) $(DOCKING_BOX_SIZE) ./main/main.cpp -g $(SRC) $(SRC_CUDA) $(LIB1) $(LIB2) $(LIB3) $(LIB_PATH) $(MACRO) $(OPTION) -DBUILD_KERNEL_FROM_SOURCE 
clean:
	rm Vina-GPU-2-1-CUDA
