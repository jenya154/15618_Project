For running the Base Code, we will compile using the following command
An executable will be created after running this

gcc baseCode.c -o baseCode.x -std=c99

./baseCode.x 256 3 where 256 is the size of the matrix aand 3 is the size of the filter

*************************************************************************************************
For running the CUDA file, we will compile it using the nvcc compiler.
An executable will be created after running this

nvcc -arch=compute_70 -o ./2d 2d.cu

./executable will run the program and display the output

*************************************************************************************************
For running the MPI file, we will compile it using the command
An executable will be created after running this

 mpicc convMPI.c -o convMPI.x -std=c99
 
 To run the file
 mpirun -np 2 ./convMPI.x where 2 is the number of processors
