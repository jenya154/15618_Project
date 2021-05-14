#include <cassert>
#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include "rdtsc.h"
#include "CycleTimer.h"

// 3 x 3 convolutional mask
#define MASK_DIM 3

__global__ void convolution_2d(float* mask, float *matrix, float *result, int matrix_size, int result_size, int dimensions) {
 
      // Calculate the global thread positions in the result array
      int row = blockIdx.y * blockDim.y + threadIdx.y;
      int col = blockIdx.x * blockDim.x + threadIdx.x;

     // Temp value for accumulating the result
      int temp = 0;

    
    // Iterate over all the rows
    for (int i = 0; i < MASK_DIM; i++) {
        // Go over each column
        for (int j = 0; j < MASK_DIM; j++) {
            //matrix[(((i * sizeMatrix)+ j)*dimension) + d + kx + ky
            // Accumulate result
            temp += matrix[(row * matrix_size)+col+i+j] * mask[(i * MASK_DIM + j)];
        }
    }


     __syncthreads();
     // Write back the result
     result[(row * result_size + col)] = temp;
  
}

// Initializes an n x n input matrix with value of 1
void init_input(float *m, int n, int dimensions) {

    for(int i = 0; i < n * n * dimensions ; ++i) {
        m[i] = 1.0;     
}
}

// Initializes filter with value of 2
void init_mask(float *m, int n, int dimensions) {
    for (int i = 0; i < n * n * dimensions; ++i) {
        m[i] = 2.0;   
    }
  }

int main() {
  
  //Dimensions of the input matrix
  int N = 8;

  //Number of channels
  int dimensions = 8;

  long long sum1 = 0;
  tsc_counter t0, t1;

  // Size of the matrix (in bytes)
  size_t bytes_n = dimensions*N * N * sizeof(float);

  //Size of the result matrix
  size_t result_size = N - MASK_DIM + 1;

  //Total number of bytes in the result array
  size_t bytes_result = dimensions*result_size * result_size *sizeof(float);

  // Allocate the matrix and initialize it
  float *matrix = new float[dimensions*N * N];
  float *result = new float[dimensions*result_size * result_size];
  init_input(matrix, N, dimensions);

  // Size of the mask in bytes
  size_t bytes_m = dimensions*MASK_DIM * MASK_DIM * sizeof(float);

  // Allocate the mask and initialize it
  float *h_mask = new float[dimensions*MASK_DIM * MASK_DIM];
  init_mask(h_mask, MASK_DIM, dimensions);

  // Allocate device memory
  float *d_matrix;
  float *d_result;
  float* d_mask;
  cudaMalloc(&d_matrix, bytes_n);
  cudaMalloc(&d_result, bytes_result);
  cudaMalloc(&d_mask, bytes_m);


  // Calculate grid dimensions
  int THREADS = result_size;
  int BLOCKS = dimensions;//(N + THREADS - 1) / THREADS;

  // Dimension launch arguments
  dim3 block_dim(THREADS, THREADS);
  dim3 grid_dim(BLOCKS,BLOCKS);

  RDTSC(t0);

  // Copy data to the device
  cudaMemcpy(d_matrix, matrix, bytes_n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask, h_mask, bytes_m, cudaMemcpyHostToDevice);

  // Perform 2D convolution
  convolution_2d<<<grid_dim, block_dim>>>(d_mask, d_matrix, d_result, N, result_size, dimensions);
  
  // Copy the result back to the CPU
  cudaMemcpy(result, d_result,bytes_result , cudaMemcpyDeviceToHost);

  RDTSC(t1);
  sum1 += (COUNTER_DIFF(t1, t0, CYCLES));

  printf("Average time: %lf cycles\n", ((double) (sum1 / ((double) runs))));

 //Printing output
for(int d = 0; d<dimensions; d++){
    for(int i = 0; i<result_size;i++){
        for(int j= 0; j<result_size;j++){
            printf("%f\t",result[ d*result_size*result_size + i*result_size + j]);
        }
        printf("\n");
    }
    printf("\n");
}

  // Free the memory we allocated
  delete[] matrix;
  delete[] result;
  delete[] h_mask;

  cudaFree(d_matrix);
  cudaFree(d_result);

  return 0;
}
