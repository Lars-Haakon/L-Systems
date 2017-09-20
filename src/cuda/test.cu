#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "test.cuh"

void PrintCudaInfo()
{
    int n_devices;
    cudaGetDeviceCount(&n_devices);
    printf("Number of CUDA devices: %d\n", n_devices);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    printf("CUDA device name: %s\n" , device_prop.name);
}
