#include    <wb.h>

#define wbCheck(stmt)                                       \
    do {                                                    \
        cudaError_t err = stmt;                             \
        if (err != cudaSuccess) {                           \
            wbLog(ERROR, "Failed to run stmt ", #stmt);     \
            return -1;                                      \
        }                                                   \
    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width / 2
#define TILE_WIDTH  16
#define SIZE        (TILE_WIDTH + Mask_width - 1)

//@@ INSERT CODE HERE
__global__
void convolution2D (float * I, const float * __restrict__ M, float * P,
        int channels, int width, int height)
{
    __shared__ float N_ds[SIZE][SIZE];

    int bx = blockIdx.x,  by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    for (int k = 0; k < channels; ++k) {
        int dest  = ty * TILE_WIDTH + tx;
        int destX = dest % SIZE;
        int destY = dest / SIZE;
        int srcY  = by * TILE_WIDTH + destY - Mask_radius;
        int srcX  = bx * TILE_WIDTH + destX - Mask_radius;
        int src   = (srcY * width + srcX) * channels + k;

        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destY][destX] = I[src];
        else
            N_ds[destY][destX] = 0.0;

        dest  = ty * TILE_WIDTH + tx + TILE_WIDTH * TILE_WIDTH;
        destY = dest / SIZE;
        destX = dest % SIZE;
        srcY  = by * TILE_WIDTH + destY - Mask_radius;
        srcX  = bx * TILE_WIDTH + destX - Mask_radius;
        src   = (srcY * width + srcX) * channels + k;

        if (destY < SIZE) {
            if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
                N_ds[destY][destX] = I[src];
            else
                N_ds[destY][destX] = 0.0;
        }
        __syncthreads();

        float accum = 0;
        for (int y = 0; y < Mask_width; ++y)
            for (int x = 0; x < Mask_width; ++x)
                accum += N_ds[ty + y][tx + x] * M[y * Mask_width + x];

        int y = by * TILE_WIDTH + ty;
        int x = bx * TILE_WIDTH + tx;
        if (y < height && x < width)
            P[(y * width + x) * channels + k] = min(max(accum, 0.0), 1.0);

        __syncthreads();
    }
}

int main (int argc, char * argv[ ])
{
    wbArg_t arg;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    arg = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(arg, 0);
    inputMaskFile = wbArg_getInputFile(arg, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData, hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData, hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");

    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
    dim3 dimGrid(ceil((float) imageWidth / TILE_WIDTH),
                 ceil((float) imageHeight / TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    convolution2D<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceMaskData,
            deviceOutputImageData, imageChannels, imageWidth, imageHeight);
    wbTime_stop(Compute, "Doing the computation on the GPU");

    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData, deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(arg, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
