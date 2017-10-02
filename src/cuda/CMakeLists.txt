#set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_52,code=sm_52)

set(CUDA_VERBOSE_BUILD ON)

cuda_add_library(cudalib cudainfo.cu STATIC)
#target_link_libraries(cudalib ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cudadevrt_LIBRARY})
