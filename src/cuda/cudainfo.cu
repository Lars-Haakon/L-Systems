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

__global__ void HillisSteeleScan(float *g_idata, float *g_odata, int n)
{
    extern __shared__ float temp[]; // allocated on invocation
    int tId = threadIdx.x;
    int pout = 0, pin = 1;
    // Load input into shared memory.
    // This is exclusive scan, so shift right by one
    // and set first element to 0
    //temp[pout*n + tId] = g_idata[tId]; // inclusive
    temp[pout*n + tId] = (tId > 0) ? g_idata[tId-1] : 0; // exclusive
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

    g_odata[tId] = temp[pout*n+tId]; // write output
}

__global__ void fill(float* f)
{
    f[threadIdx.x*3] = threadIdx.x*3.0f;
    f[threadIdx.x*3+1] = 0.0f;
    f[threadIdx.x*3+2] = 0.0f;
}

void FillData(unsigned int vbo)
{
    /*texture<float> texture_reference(2);
    const void* tobereference;
    cudaBindTexture( NULL, texture_reference, tobereference, 0 );


    cudaUnbindTexture ( texture_reference );*/

    struct cudaGraphicsResource* cudaResource;
    cudaGraphicsGLRegisterBuffer(&cudaResource, vbo, cudaGraphicsMapFlagsNone);

    cudaGraphicsMapResources(1, &cudaResource, 0);
    float* positions;
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes, cudaResource);

    int n = 8;
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

    dim3 numThreadsPerBlock(n);
    dim3 numBlocks(1);
    scan<<<numBlocks, numThreadsPerBlock, 2*n*sizeof(float)>>>(device_in, device_out, n);

    float* host_out = (float*) malloc(n*sizeof(float));
    cudaMemcpy(host_out, device_out, n*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < n; i++)
    {
        printf("%.2f\n", host_out[i]);
    }

    cudaGraphicsUnmapResources(1, &cudaResource, 0);

    cudaGraphicsUnregisterResource(cudaResource);
}
