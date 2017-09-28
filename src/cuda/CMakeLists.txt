set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_52,code=sm_52)

cuda_add_library (cudalib STATIC cudainfo.cu)
