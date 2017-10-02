#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cublas_v2.h>

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

__global__ void HillisSteeleScan(float* transform, int *count, int n)
{
    extern __shared__ int temp[]; // allocated on invocation

    int tId = threadIdx.x;
    int pout = 0, pin = 1;
    // Load input into shared memory.
    // This is exclusive scan, so shift right by one
    // and set first element to 0
    temp[tId] = count[tId]; // inclusive
    //temp[tId] = (tId > 0) ? in[tId-1] : 0; // exclusive
    __syncthreads();
    for (int offset = 1; offset < n; offset *= 2)
    {
        pout = 1 - pout; // swap double buffer indices
        pin = 1 - pout;
        if (tId >= offset)
            temp[pout*n+tId] = temp[pin*n+tId] + temp[pin*n+tId - offset];
        else
            temp[pout*n+tId] = temp[pin*n+tId];

        __syncthreads();
    }

    count[tId] = temp[pout*n+tId]; // write output
}

__global__ void Count(float* device_lookUpTable, char* device_module, float* device_transform, int* device_count)
{
    cublasHandle_t cnpHandle;
    cublasStatus_t status = cublasCreate(&cnpHandle);

    int bId = blockIdx.x;
    int tId = threadIdx.x;

    device_transform[bId*16 + tId] = device_lookUpTable[device_module[bId]*16 + tId];

    if(tId == 0) // only need 1 thread to set value to avoid conflicts
    {
        device_count[bId] = 0;

        if(device_module[bId] == 'F')
        {
            device_count[bId] = 1;
        }
    }
}

int FillData(float* lookUpTable, int lookUpTableSize, const char* module, int moduleLength)
{
    // move lookUpTable to device
    float* device_lookUpTable = 0;
    cudaMalloc((void**)&device_lookUpTable, lookUpTableSize);
    cudaMemcpy((void*)device_lookUpTable, (void*)lookUpTable, lookUpTableSize, cudaMemcpyHostToDevice);

    char* device_module = 0;
    cudaMalloc((void**)&device_module, moduleLength);
    cudaMemcpy((void*)device_module, (void*)module, moduleLength, cudaMemcpyHostToDevice);

    float* device_transform = 0;
    cudaMalloc((void**)&device_transform, 2*moduleLength*16*sizeof(float)); // multiply by 2 to use this as a double buffer

    int* device_count = 0;
    cudaMalloc((void**)&device_count, moduleLength * sizeof(int));

    dim3 numThreadsPerBlock(16);
    dim3 numBlocks(moduleLength);
    Count<<<numBlocks, numThreadsPerBlock>>>(device_lookUpTable, device_module, device_transform, device_count);

    dim3 numThreadsPerBlock2(moduleLength);
    dim3 numBlocks2(1);
    HillisSteeleScan<<<numBlocks2, numThreadsPerBlock2, 2*moduleLength*sizeof(int)>>>(device_transform, device_count, moduleLength);

    /*float* host_out = (float*) malloc(16*sizeof(float));
    cudaMemcpy(host_out, device_transform+16, 16*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 16; i++)
    {
        printf("%.2f\n", host_out[i]);
    }*/

    // get the last value
    int size = -1;
    cudaMemcpy(&size, device_count + moduleLength-1, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_count);
    cudaFree(device_module);
    cudaFree(device_lookUpTable);

    return size;

    /*int n = 512; // 512 is max for the HillisSteeleScan
    float* host_in = (float*)malloc(n*sizeof(float));
    for(int i = 0; i < n; i++)
    {
        host_in[i] = i+1;
    }

    float* device_in = 0;
    float* device_out = 0;
    cudaMalloc((void**)&device_in, n*sizeof(float));
    cudaMalloc((void**)&device_out, n*sizeof(float));

    cudaMemcpy((void*)device_in, (void*)host_in, n*sizeof(float), cudaMemcpyHostToDevice);

    HillisSteeleScan<<<numBlocks, numThreadsPerBlock, 2*n*sizeof(float)>>>(device_in, device_out, n);

    float* host_out = (float*) malloc(n*sizeof(float));
    cudaMemcpy(host_out, device_out, n*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < n; i++)
    {
        printf("%.2f\n", host_out[i]);
    }*/
}

void FillVBO(unsigned int vbo)
{
    struct cudaGraphicsResource* cudaResource;
    cudaGraphicsGLRegisterBuffer(&cudaResource, vbo, cudaGraphicsMapFlagsNone);

    cudaGraphicsMapResources(1, &cudaResource, 0);
    float* positions;
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes, cudaResource);













    cudaGraphicsUnmapResources(1, &cudaResource, 0);
    cudaGraphicsUnregisterResource(cudaResource);
}
