#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rdtsc.h"
#include <immintrin.h>
#include <pthread.h>
#include <mpi.h>

void printOutput(int sizeResult, int dimension, float* result){
    for(int d = 0; d<dimension; d++){
        for(int i = 0; i<sizeResult;i++){
            for(int j= 0; j<sizeResult;j++){
                printf("%f\t",result[ i*sizeResult*dimension + j*dimension + d]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void correctness(int sizeResult, int sizeFilter, int dimension, float* result){
    for(int d = 0; d<dimension; d++){
        for(int i = 0; i<sizeResult ;i++){
            for(int j= 0; j<sizeResult;j++){
                if (result[ i*sizeResult*dimension + j*dimension + d] != sizeFilter*sizeFilter*2){
                    printf("%d %d %d\t", i,j,d);
                }    
            }
        }
    }
}
void kernel( int sizeFilter, int sizeMatrix, int sizeResult, int dimStart, int dimEnd, int dimension, float* restrict matrix, float* restrict result, float* restrict filter ) {
    /*
     * Assumptions:
     * matrix - stored row-wise
     * res - stored row-wise
     * filter - stored row-wise
    */

    for(int d = dimStart; d < dimEnd; d++){
        for (int i = 0; i < sizeResult; i++) {
            for (int j = 0; j < sizeResult; j++) {
                for (int kx = 0; kx < sizeFilter; kx++) {
                    for (int ky = 0; ky < sizeFilter; ky++) {
                        result[ d*sizeResult*sizeResult + i*sizeResult + j] += matrix[ d*sizeMatrix*sizeMatrix + (i * sizeMatrix)+ j + kx + ky] * filter[d*sizeFilter*sizeFilter + kx * sizeFilter + ky];
                    }
                }  
            }
        }
    }
    
}

int main(int argc, char **argv){
    int sizeMatrix = 512;
    int sizeFilter = 3;
    int runs = 3;
    int dimension = 64;

    float *matrix ;
    float *filter ;
    float *result ;

    int padding = 0;
    int strides = 1;
    long long sum1 = 0;
    tsc_counter t0, t1;

    int myid, size;
    RDTSC(t0);
    MPI_Init(&argc, &argv);

    int root = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    //printf("Size: %d\n",size);
    int perProcessor = dimension/size;

    int sizeResult = (((sizeMatrix - sizeFilter + 2 * padding) / strides) + 1);
    // printf("size Result: %d\n", sizeResult);


    float *resultPartial;
    float *matrixPartial;
    float *filterPartial;

    int valResult = sizeResult * sizeResult * perProcessor;
    int valFilter = sizeFilter * sizeFilter * perProcessor;
    int valMatrix = sizeMatrix * sizeMatrix * perProcessor;

    posix_memalign((void**) &matrixPartial, 64, perProcessor * sizeMatrix * sizeMatrix * sizeof(float));
    posix_memalign((void**) &filterPartial, 64, perProcessor * sizeFilter * sizeFilter * sizeof(float));
    posix_memalign((void**) &resultPartial, 64, perProcessor * sizeResult * sizeResult * sizeof(float));

    if(myid == root){
        posix_memalign((void**) &matrix, 64, dimension * sizeMatrix * sizeMatrix * sizeof(float));
        posix_memalign((void**) &filter, 64, dimension * sizeFilter * sizeFilter * sizeof(float));
        posix_memalign((void**) &result, 64, dimension * sizeResult * sizeResult * sizeof(float));
    
        for (int i = 0; i < sizeFilter * sizeFilter * dimension; ++i) {
            filter[i] = 2.0;   
        }

        for(int i = 0; i < sizeMatrix * sizeMatrix * dimension ; ++i) {
            matrix[i] = 1.0;     
        }

        for(int i = 0; i < dimension * sizeResult * sizeResult;++i) {
            result[i] = 0.0;
        }
        
    }
    
    //scatter respective matrix, result and filter position 
    MPI_Scatter(result, valResult, MPI_FLOAT, resultPartial, valResult, MPI_FLOAT, root, MPI_COMM_WORLD  );
    MPI_Scatter(matrix, valMatrix, MPI_FLOAT, matrixPartial, valMatrix, MPI_FLOAT, root, MPI_COMM_WORLD  );    
    MPI_Scatter(filter, valFilter, MPI_FLOAT, filterPartial, valFilter, MPI_FLOAT, root, MPI_COMM_WORLD  );
    

    for(long i = 0; i <size ;i++){
        if(myid == i){
            int dimStart = 0;
            int dimEnd = dimension/size;
            tsc_counter t2, t3;
            RDTSC(t2);
            kernel(sizeFilter,sizeMatrix,sizeResult,dimStart,dimEnd,dimension,matrixPartial,resultPartial,filterPartial);
            RDTSC(t3);
            //sum1 += (COUNTER_DIFF(t3, t2, CYCLES));
            //printf("%lf + ", ((double) (sum1 / ((double) runs))));
        }   
    }

    
    MPI_Gather(resultPartial, valResult, MPI_FLOAT, result, valResult, MPI_FLOAT, root, MPI_COMM_WORLD);
    
    MPI_Finalize();
    RDTSC(t1);

    if(myid == root){
        sum1 += (COUNTER_DIFF(t1, t0, CYCLES));
        printf("Average time: %lf cycles\n", ((double) (sum1 / ((double) runs))));
        correctness(sizeResult, sizeFilter, dimension, result);
        //printOutput(sizeResult, dimension, result);
    }  
    
    free(matrix);
    free(result);
    free(filter);
}


 /*
    mpicc convMPI.c -o convMPI.x -std=c99
 */
