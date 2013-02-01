#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                       \
    do {                                                    \
        cudaError_t err = stmt;                             \
        if (err != cudaSuccess) {                           \
            wbLog(ERROR, "Failed to run stmt ", #stmt);     \
            return -1;                                      \
        }                                                   \
    } while(0)

__global__
void fixup (float *input, float *temp, int len)
{
    if (blockIdx.x == 0) return;

    unsigned int start = 2 * blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (start < len)
        input[start] += temp[blockIdx.x - 1];
    if (start + BLOCK_SIZE < len)
        input[start + BLOCK_SIZE] += temp[blockIdx.x - 1];
}

__global__
void scan (float * input, float * output, float *temp, int len)
{
    __shared__ float scarray[BLOCK_SIZE << 1];

    unsigned int t     = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * BLOCK_SIZE + t;

    scarray[t] = start < len ? input[start] : 0.0;
    scarray[BLOCK_SIZE + t] = (start + BLOCK_SIZE) < len ?
                              input[start + BLOCK_SIZE]  : 0.0;
    __syncthreads();

    int index;
    for (int stride = 1; stride <= BLOCK_SIZE; stride <<= 1) {
        index = (t + 1) * stride * 2 - 1;

        if (index < 2 * BLOCK_SIZE)
            scarray[index] += scarray[index - stride];

        __syncthreads();
    }

    for (int stride = BLOCK_SIZE >> 1; stride; stride >>= 1) {
        index = (t + 1) * stride * 2 - 1;

        if (index + stride < 2 * BLOCK_SIZE)
            scarray[index + stride] += scarray[index];

        __syncthreads();
    }

    if (start < len)
        output[start] = scarray[t];
    if (start + BLOCK_SIZE < len)
        output[start + BLOCK_SIZE] = scarray[BLOCK_SIZE + t];

    if (temp != NULL && t == 0)
        temp[blockIdx.x] = scarray[2 * BLOCK_SIZE - 1];
}

int main (int argc, char ** argv)
{
    wbArg_t   args;
    float   * hostInput;                // The input 1D list
    float   * hostOutput;               // The output list
    float   * deviceInput;
    float   * deviceOutput;
    float   * deviceTemp;
    float   * deviceTempScanned;
    int       numElements;              // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    cudaHostAlloc(&hostOutput, numElements * sizeof(float), cudaHostAllocDefault);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void **) &deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void **) &deviceOutput, numElements*sizeof(float)));
    cudaMalloc(&deviceTemp, (BLOCK_SIZE << 1) * sizeof(float));
    cudaMalloc(&deviceTempScanned, (BLOCK_SIZE << 1) * sizeof(float));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 dimGrid(ceil((float) numElements / (BLOCK_SIZE << 1)), 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan on the device
    scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, deviceTemp, numElements);
    cudaDeviceSynchronize();
    scan<<<dim3(1, 1, 1), dimBlock>>>(deviceTemp, deviceTempScanned, NULL, BLOCK_SIZE << 1);
    cudaDeviceSynchronize();
    fixup<<<dimGrid, dimBlock>>>(deviceOutput, deviceTempScanned, numElements);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFree(deviceTemp);
    cudaFree(deviceTempScanned);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    cudaFreeHost(hostOutput);

    return 0;
}

