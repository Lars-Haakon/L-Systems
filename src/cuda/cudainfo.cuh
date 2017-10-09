#ifndef CUDAINFO_CUH
#define CUDAINFO_CUH

extern void PrintCudaInfo();
extern int FillData(float* lookUpTable, int lookUpTableSize, const char* module, int moduleLength);
extern void FillBuffers(unsigned int vbo, unsigned int ibo, int moduleLength);

#endif
