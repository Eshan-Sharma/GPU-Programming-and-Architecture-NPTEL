#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "device_launch_parameters.h"
#define LOG_INPUT if(0)
#define LOG_OUTPUT if(1)
#define LOG if(0)


__global__ void matrix_add(float* A, float* B, float* C, int M, int N)
{
    // Complete the kernel code snippet
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * Host main routine
 */
void vecAdd(float* A, float* B, float* C, int M, int N)
{
    //host program
    int size = N * sizeof(float);
    float* d_A = NULL, * d_B = NULL, * d_C = NULL;
    cudaError_t err = cudaSuccess;
}

void print_matrix(float* A, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            printf("%.2f ", A[i * n + j]);
        printf("\n");
    }

}

int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int N = 0;
    scanf("%d", &N);

    int t; //number of test cases
    scanf("%d", &t);
    while (t--)
    {
        int m, n;
        scanf("%d %d", &m, &n);
        size_t size = m * n * sizeof(float);
        LOG printf("[Matrix Add of two matrices ]\n");

        // Allocate the host input vector A
        float* h_A = (float*)malloc(size);

        // Allocate the host input vector B
        float* h_B = (float*)malloc(size);

        // Allocate the host output vector C
        float* h_C = (float*)malloc(size);

        // Verify that allocations succeeded
        if (h_A == NULL || h_B == NULL || h_C == NULL)
        {
            fprintf(stderr, "Failed to allocate host vectors!\n");
            exit(EXIT_FAILURE);
        }

        // Initialize the host input vectors

        for (int i = 0; i < n * m; ++i)
        {
            scanf("%f", &h_A[i]);
            scanf("%f", &h_B[i]);

        }


        // Allocate the device input vector A
        float* d_A = NULL;
        err = cudaMalloc((void**)&d_A, size);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Allocate the device input vector B
        float* d_B = NULL;
        err = cudaMalloc((void**)&d_B, size);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Allocate the device output vector C
        float* d_C = NULL;
        err = cudaMalloc((void**)&d_C, size);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }


        // Copy the host input vectors A and B in host memory to the device input vectors in
        // device memory
        err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector A from host to device (error code %s)! \n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector B from host to device (error code %s)! \n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        // initialize blocksPerGrid and threads Per Block
        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        printf("CUDA kernel launch with %d blocks of %d threads \n", threadsPerBlock, blocksPerGrid);
        vecAdd <<blocksPerGrid, threadsPerBlock >>(d_A, d_B, d_C, n);
        err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }


        matrix_add << blocksPerGrid, threadsPerBlock >>  (d_A, d_B, d_C, m, n);
        err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Copy the device result vector in device memory to the host result vector
        // in host memory.
        printf("Copy output data from the output device to the host memory\n", cudaGetErrorString(err));
        err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector C from device to host(error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }


        // Verify that the result vector is correct
        for (int i = 0; i < n * m; ++i)
        {
            if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
            {
                fprintf(stderr, "Result verification failed at element %d!\n", i);
                exit(EXIT_FAILURE);
            }
        }

        LOG printf("Test PASSED\n");

        // Free device global memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // Free host memory
        free(d_A);
        free(d_B);
        free(d_C);

        err = cudaDeviceReset();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        print_matrix(h_C, m, n);

        LOG printf("Done\n");
    }
    return 0;
}
