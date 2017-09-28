#ifndef CUDAINFO_CUH
#define CUDAINFO_CUH

extern void PrintCudaInfo();
extern int FillData(float* lookUpTable, int lookUpTableSize, const char* module, int moduleLength);
extern void FillVBO(unsigned int vbo);

#endif
