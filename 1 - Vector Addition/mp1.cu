#include "wb.h"

#define check(A, M) { if (!A) wbLog(ERROR, M); }

__global__
void vecAdd (float * in1, float * in2, float * out, int len)
{
    //@@ Insert code to implement vector addition here
    int i = blockIdx.x * blockDiim.x + threadIdx.x;
    if (i < len) out[i] = in1[i] + in2[i];
}

int main (int argc, char ** argv)
{
    wbArg_t args;
    cudaError_t error;
    int inputSize;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);
    inputSize = inputLength * sizeof(float);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    error = cudaMalloc((void **) &deviceInput1, inputSize);
    check(error == cudaSuccess, "Couldn't allocate memory of Input1");
    error = cudaMalloc((void **) &deviceInput2, inputSize);
    check(error == cudaSuccess, "Couldn't allocate memory of Input2");
    error = cudaMalloc((void **) &deviceOutput, inputSize);
    check(error == cudaSuccess, "Couldn't allocate memory of Output");

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    error = cudaMemcpy(deviceInput1, hostInput1, inputSize, cudaMemcpyHostToDevice);
    check(error == cudaSuccess, "Couldn't copy Input1 data to the GPU");
    error = cudaMemcpy(deviceInput2, hostInput2, inputSize, cudaMemcpyHostToDevice);
    check(error == cudaSuccess, "Couldn't copy Input2 data to the GPU");

    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid(ceil(inputLength / 256.f), 1, 1);
    dim3 DimBlock(256, 1, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    vecAdd<<<DimGrid, DimBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    error = cudaMemcpy(hostOutput, deviceOutput, inputSize, cudaMemcpyDeviceToHost);
    check(error == cudaSuccess, "Couldn't copy Output memory to CPU");

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}

