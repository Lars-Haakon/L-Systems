#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "cudainfo.cuh"

extern void PrintCudaInfo()
{
    int n_devices;
    cudaGetDeviceCount(&n_devices);
    printf("Number of CUDA devices: %d\n", n_devices);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    printf("CUDA device name: %s\n" , device_prop.name);
}

__global__ void Fill(char* device_module, float* transform, int *count)
{
    int tId = threadIdx.x;

    if(device_module[tId] == 'F')
    {
        
    }
}

__global__ void HillisSteeleScan(float* transform, int *count, int n)
{
    extern __shared__ int temp[]; // allocated on invocation

    int tId = threadIdx.x;
    int pout = 0, pin = 1;
    // Load input into shared memory.
    // This is exclusive scan, so shift right by one
    // and set first element to 0
    temp[pout*n + tId] = count[tId]; // inclusive
    //temp[pout*n + tId] = (tId > 0) ? in[tId-1] : 0; // exclusive
    __syncthreads();
    for (int offset = 1; offset < n; offset *= 2)
    {
        pout = 1 - pout; // swap double buffer indices
        pin = 1 - pout;

        temp[pout*n+tId] = temp[pin*n+tId] + ((tId >= offset)?temp[pin*n+tId - offset]:0);

        float* A = &transform[16*(pin*n + tId)];
        float* B = &transform[16*(pin*n + tId - offset)];
        float* C = &transform[16*(pout*n + tId)];

        for (int i = 0; i < 4; i++)
    	{
    		for (int j = 0; j < 4; j++)
    		{
    			C[i*4 + j] =	A[i*4+0] * ((tId >= offset)?B[0*4+j]:0) +
    							A[i*4+1] * ((tId >= offset)?B[1*4+j]:0) +
    							A[i*4+2] * ((tId >= offset)?B[2*4+j]:0) +
    							A[i*4+3] * ((tId >= offset)?B[3*4+j]:0);
    		}
        }

        __syncthreads();
    }

    count[tId] = temp[pout*n+tId]; // write output
    float* A = &transform[16*(pin*n + tId)];
    float* C = &transform[16*tId];
    for(int i = 0; i < 16; i++)
        C[i] = A[i];
}

__global__ void Count(float* device_lookUpTable, char* device_module, float* device_transform, int* device_count)
{
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

float* device_lookUpTable = 0;
char* device_module = 0;
float* device_transform = 0;
int* device_count = 0;

extern int FillData(float* lookUpTable, int lookUpTableSize, const char* module, int moduleLength)
{
    // move lookUpTable to device
    cudaMalloc((void**)&device_lookUpTable, lookUpTableSize);
    cudaMemcpy((void*)device_lookUpTable, (void*)lookUpTable, lookUpTableSize, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&device_module, moduleLength);
    cudaMemcpy((void*)device_module, (void*)module, moduleLength, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&device_transform, 2*moduleLength*16*sizeof(float)); // multiply by 2 to allocate a double buffer in shared memory

    cudaMalloc((void**)&device_count, moduleLength * sizeof(int));

    dim3 numThreadsPerBlock(16);
    dim3 numBlocks(moduleLength);
    Count<<<numBlocks, numThreadsPerBlock>>>(device_lookUpTable, device_module, device_transform, device_count);

    dim3 numThreadsPerBlock2(moduleLength);
    dim3 numBlocks2(1);
    HillisSteeleScan<<<numBlocks2, numThreadsPerBlock2, 2*moduleLength*sizeof(int)>>>(device_transform, device_count, moduleLength);

    /*float* host_matrix = (float*)malloc(16*sizeof(float));
    cudaMemcpy(host_matrix, device_transform+7*16, 16*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 4; i++)
    {
        for(int j = 0; j < 4; j++)
        {
            printf("%.2f ", host_matrix[i*4+j]);
        }
        printf("\n");
    }*/
    /*int* host_out = (int*) malloc(moduleLength*sizeof(int));
    cudaMemcpy(host_out, device_count, moduleLength*sizeof(int), cudaMemcpyDeviceToHost);
    for(int i = 0; i < moduleLength; i++)
    {
        printf("%d\n", host_out[i]);
    }*/

    // get the last value
    int size = -1;
    cudaMemcpy(&size, device_count + moduleLength-1, sizeof(int), cudaMemcpyDeviceToHost);

    return size;
}

extern void FillVBO(unsigned int vbo, int moduleLength)
{
    struct cudaGraphicsResource* cudaResource;
    cudaGraphicsGLRegisterBuffer(&cudaResource, vbo, cudaGraphicsMapFlagsNone);

    cudaGraphicsMapResources(1, &cudaResource, 0);
    float* positions;
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes, cudaResource);

    dim3 numThreadsPerBlock(moduleLength);
    dim3 numBlocks(1);
    Fill<<<numBlocks, numThreadsPerBlock>>>(device_module, device_transform, device_count);

    cudaFree(device_lookUpTable);
    cudaFree(device_module);
    cudaFree(device_transform);
    cudaFree(device_count);

    cudaGraphicsUnmapResources(1, &cudaResource, 0);
    cudaGraphicsUnregisterResource(cudaResource);
}
