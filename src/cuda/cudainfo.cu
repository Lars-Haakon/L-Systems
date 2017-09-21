#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "cudainfo.cuh"

void PrintCudaInfo()
{
    int n_devices;
    cudaGetDeviceCount(&n_devices);
    printf("Number of CUDA devices: %d\n", n_devices);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    printf("CUDA device name: %s\n" , device_prop.name);
}

__global__ void fill(float* f)
{
    f[threadIdx.x*3] = threadIdx.x*3.0f;
    f[threadIdx.x*3+1] = 0.0f;
    f[threadIdx.x*3+2] = 0.0f;
}

void FillData(unsigned int vbo)
{
    struct cudaGraphicsResource* cudaResource;
    cudaGraphicsGLRegisterBuffer(&cudaResource, vbo, cudaGraphicsMapFlagsNone);

    cudaGraphicsMapResources(1, &cudaResource, 0);
    float* positions;
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes, cudaResource);

    dim3 dimBlock( 8, 1 );
	dim3 dimGrid( 1, 1 );
    fill<<<dimGrid, dimBlock>>>(positions);

    cudaGraphicsUnmapResources(1, &cudaResource, 0);

    cudaGraphicsUnregisterResource(cudaResource);
}
