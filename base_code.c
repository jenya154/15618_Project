#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rdtsc.h"
#include <immintrin.h>

void kernel( int sizeKernel, int sizeMatrix, int sizeResult, int dimension, float* restrict matrix, float* restrict result, float* restrict filter ) {
    /*
     * Assumptions:
     * matrix - stored row-wise
     * res - stored row-wise
     * filter - stored row-wise
    */


    for(int d = 0; d < dimension; d++){
        for (int i = 0; i < sizeResult; i++) {
            for (int j = 0; j < sizeResult; j++) {
                for (int kx = 0; kx < sizeKernel; kx++) {
                    for (int ky = 0; ky < sizeKernel; ky++) {
                         result[((i*sizeResult + j)*dimension) + d] += matrix[d*sizeMatrix*sizeMatrix + (i * sizeMatrix)+ j + kx + ky] * 
                         filter[d*sizeKernel*sizeKernel + kx * sizeKernel + ky];
                    }
                }  
            }
        }
    }



}

int main(int argc, char **argv){
    int sizeMatrix = atoi(argv[1]);
    int sizeKernel = atoi(argv[2]);
    int  runs = 3;
    int dimension = 8;

    float *matrix ;
    float *filter ;
    float *result ;

    int padding = 0;
    int strides = 1;
    long long sum1 = 0;
    tsc_counter t0, t1;

    int sizeResult = (((sizeMatrix - sizeKernel + 2 * padding) / strides) + 1);
    printf("size Result: %d\n", sizeResult);

    posix_memalign((void**) &matrix, 64, dimension * sizeMatrix * sizeMatrix * sizeof(float));
    posix_memalign((void**) &filter, 64, dimension * sizeKernel * sizeKernel * sizeof(float));
    posix_memalign((void**) &result, 64, dimension * sizeResult * sizeResult * sizeof(float));

    for (int i = 0; i < sizeKernel * sizeKernel * dimension; ++i) {
        filter[i] = 2.0;   
    }

    for(int i = 0; i < sizeMatrix * sizeMatrix * dimension ; ++i) {
            matrix[i] = 1.0;     
    }

    for(unsigned int i = 0; i != runs; ++i) {
        for(int i = 0; i < dimension * sizeResult * sizeResult;++i) {
            result[i] = 0.0;
        }
        RDTSC(t0);
        kernel(sizeKernel,sizeMatrix,sizeResult,dimension,matrix,result,filter);
        
        RDTSC(t1);
        sum1 += (COUNTER_DIFF(t1, t0, CYCLES));

    }
    printf("Average time: %lf cycles\n", ((double) (sum1 / ((double) runs))));

/**
 * To print out the result
*/
/*for(int d = 0; d<dimension; d++){
    for(int i = 0; i<sizeResult;i++){
        for(int j= 0; j<sizeResult;j++){
            printf("%f\t",result[ d*sizeResult*sizeResult + i*sizeResult + j]);
        }
        printf("\n");
    }
    printf("\n");
} */
 
    
    free(matrix);
    free(result);
    free(filter);
}


 /*
 	gcc -O3 -Wall -mavx -mavx2 -mfma base-1D.c -o base-1D.x -std=c99
	./base-1D.x 1024 100
 */
