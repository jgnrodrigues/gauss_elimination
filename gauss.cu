/////////////////////////////////////////////////////////////////////
//  RESOLVING SYSTEM OF LINEAR EQUATIONS USING GAUSSIAN ELIMINATION
//  CUDA
//
//  João Rodrigues
//  MEC: 71771
//
//  algorithm based on http://www.bragitoff.com/2015/09/c-program-for-gauss-elimination-for-solving-a-system-of-linear-equations/
/////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "common/common.h"
#include <cuda_runtime.h>

//cpu
void gauss_cpu(double matrix[], double result[], int n)
{
	int i,j,k;
    int nvar = n + 1;
	//Pivotisation
	for (i=0;i<n;i++)
	{                    
        for (k=i+1;k<n;k++)
		{
			if (matrix[nvar*i+i]<matrix[nvar*k+i])
			{
				for (j=0;j<=n;j++)
				{
					double temp=matrix[nvar*i+j];
					matrix[nvar*i+j]=matrix[nvar*k+j];
					matrix[nvar*k+j]=temp;
				}
			}
		}
	}

    ///////////////////////////
    printf("\nThe matrix after Pivotisation is:\n");
    for (i=0;i<n;i++)            //print the new matrix
    {
        for (j=0;j<=n;j++)
            printf("%f\t",matrix[nvar*i+j]);
        printf("\n");
    }    
    //////////////////////////

	//loop to perform the gauss elimination
	/*
		i -> diagonal row
		k -> rows below
		j -> collumn
	*/
	for (i=0;i<n-1;i++)
	{            
        for (k=i+1;k<n;k++)
		{
			double ratio=matrix[nvar*k+i]/matrix[nvar*i+i]; 
			for (j=0;j<=n;j++)
				matrix[nvar*k+j]=matrix[nvar*k+j]-ratio*matrix[nvar*i+j];    //make the elements below the pivot elements equal to zero or elimnate the variables
		}
	}
    /////////////////////////////
     printf("\n\nThe matrix after gauss-elimination is as follows:\n");
     for (i=0;i<n;i++)            //print the new matrix
    {
        for (j=0;j<=n;j++)
            printf("%f\t",matrix[nvar*i+j]);
        printf("\n");
    }    
    ////////////////////////////

	for (i=n-1;i>=0;i--)                //back-substitution
    {                        
        result[i]=matrix[nvar*i+n];                //make the variable to be calculated equal to the rhs of the last equation
        for (j=0;j<n;j++)
            if (j!=i)            //then subtract all the lhs values except the coefficient of the variable whose value is being calculated
                result[i]=result[i]-matrix[nvar*i+j]*result[j];
        result[i]=result[i]/matrix[nvar*i+i];            //now finally divide the rhs by the coefficient of the variable to be calculated
    }
}

//gauss elimination   CUDA kernel
__global__ void gaussOnGPU(double matrix[], const int n)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * gridDim.x * blockDim.x + ix;
    int i;
    int NVAR = n + 1;

    int k = idx / NVAR; //line of matrix wich idx belongs
    int j = idx % NVAR; //collumn of matrix wich idx belongs

    if (idx < (gridDim.x * gridDim.y * blockDim.x * blockDim.y) && k < n && j <= n)
    {
        for (i=0;i<n-1;i++)
        {   
            /*         
            for (k=i+1;k<n;k++)
            {
                double ratio=matrix[nvar*k+i]/matrix[nvar*i+i]; 
                for (j=0;j<=n;j++)
                    matrix[nvar*k+j]=matrix[nvar*k+j]-ratio*matrix[nvar*i+j];    //make the elements below the pivot elements equal to zero or elimnate the variables
            }
            */
            
            if(k > i) //compute if the k line is below of i line
            {
                double ratio = matrix[NVAR*k+i]/matrix[NVAR*i+i];
                matrix[NVAR*k+j] = matrix[NVAR*k+j]-ratio*matrix[NVAR*i+j];       //make the elements below the pivot elements equal to zero or elimnate the variables
            }
            __syncthreads(); //Synchronizing all threads before next iteration 
        }
    }
}

//GPU
void gauss_gpu(double matrix[], double result[], int n)
{
    //setup device
    int dev = 0;
    cudaDeviceProp deviceProp;

    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    size_t matrix_size = sizeof(double) * n * (n + 1);
    size_t result_size = sizeof(double) * n;

    if(matrix_size > deviceProp.totalGlobalMem)
    {
        fprintf(stderr,"Not enough memory in the cuda device\n");
        exit(1);
    }

    int i,j,k;
    int nvar = n + 1;
    double* temp = (double*)calloc((size_t)(nvar+1), (size_t)(sizeof(double)));
	//Pivotisation
	for (i=0;i<n;i++)
	{                    
        for (k=i+1;k<n;k++)
		{
			if (matrix[nvar*i+i]<matrix[nvar*k+i])
			{
                mempcpy(temp, matrix+nvar*i, sizeof(double)*nvar);
                mempcpy(matrix+nvar*i, matrix+nvar*k, sizeof(double)*nvar);
                mempcpy(matrix+nvar*k, temp, sizeof(double)*nvar);
			}
		}
	}
    free(temp);

    ///////////////////////////
    printf("\nThe matrix after Pivotisation is:\n");
    for (i=0;i<n;i++)            //print the new matrix
    {
        for (j=0;j<=n;j++)
            printf("%f\t",matrix[nvar*i+j]);
        printf("\n");
    }    

	//loop to perform the gauss elimination
	/*
		i -> diagonal row
		k -> rows below
		j -> collumn
	*/

    /*
	for (i=0;i<n-1;i++)
	{            
        for (k=i+1;k<n;k++)
		{
			double ratio=matrix[nvar*k+i]/matrix[nvar*i+i]; 
			for (j=0;j<=n;j++)
				matrix[nvar*k+j]=matrix[nvar*k+j]-ratio*matrix[nvar*i+j];    //make the elements below the pivot elements equal to zero or elimnate the variables
		}
	}
    */   

    //device memory allocation
    double *dev_matrix;
    CHECK(cudaMalloc((void **)&dev_matrix, matrix_size));

    // transfer data from host to device
    CHECK(cudaMemcpy(dev_matrix, matrix, matrix_size, cudaMemcpyHostToDevice));

    // device configurations
    int block_x = n + 1;
    int block_y = n;

    int grid_x = n + 1;
    int grid_y = n;

    //block size verification
    if(block_x > deviceProp.maxThreadsDim[0] || block_y > deviceProp.maxThreadsDim[1] || block_x * block_y > deviceProp.maxThreadsPerBlock)
    {
        fprintf(stderr,"Block too big\n");
        exit(1);
    }

    //grid size verification
    if(grid_x > deviceProp.maxGridSize[0] || grid_y > deviceProp.maxGridSize[1])
    {
        fprintf(stderr,"Grid too big\n");
        exit(1);
    }

    dim3 block (n + 1, n);
    dim3 grid  (1);

    //kernel execution
    gaussOnGPU<<<grid, block>>>(dev_matrix, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    printf("\ngaussOnGPU<<<%d, %d>>>: block: %d x %d\n", grid.x * grid.y, block.x * block.y, block.x, block.y);  

    // transfer data from device to host
    CHECK(cudaMemcpy(matrix, dev_matrix,matrix_size, cudaMemcpyDeviceToHost));

    //free device memory
    CHECK(cudaFree(dev_matrix));

    /////////////////////////////
     printf("\n\nThe matrix after gauss-elimination is as follows:\n");
     for (i=0;i<n;i++)            //print the new matrix
    {
        for (j=0;j<=n;j++)
            printf("%f\t",matrix[nvar*i+j]);
        printf("\n");
    }    
    ////////////////////////////

	for (i=n-1;i>=0;i--)                //back-substitution
    {                        
        result[i]=matrix[nvar*i+n];                //make the variable to be calculated equal to the rhs of the last equation
        for (j=0;j<n;j++)
            if (j!=i)            //then subtract all the lhs values except the coefficient of the variable whose value is being calculated
                result[i]=result[i]-matrix[nvar*i+j]*result[j];
        result[i]=result[i]/matrix[nvar*i+i];            //now finally divide the rhs by the coefficient of the variable to be calculated
    }

}

int read_file(const char* fileName,double* matrix, int* n)
{
    FILE *file;
    file=fopen(fileName, "r");

    int nvar;
    if (!fscanf(file, "%d", &nvar))
    {
         fprintf(stderr,"Can't read file");
         return -1;
    }

    if(nvar == 0 || nvar != *n)
    {
        fprintf(stderr,"Can't read nvar");
        return -1;
    }

    int i,j;

    /*matrix*/

    for(i = 0; i < nvar * (nvar+1); i++)
    {
        //Use lf format specifier, %c is for character
        if (!fscanf(file, "%lf", matrix +i )) 
            break;
    }
    /*
    for(int i = 0; i< nvar; i++)
    {
        for (int j = 0; j <= nvar; j++)
        {
            if (!fscanf(file, "%lf", &mat[(nvar+1) * i + j])) 
                break;
        }
    }
    */
    fclose(file);  

    *n = nvar;
    return 0;
}

int verification(double* result_a, double* result_b, int n)
{
    int i;
    for(i = 0; i < n; i++)
    {
        if((result_a[i] - result_b[i]) > 1e-14)
        {
            printf("Result mismatch:\n");
            printf("i: %d\tCPU: %f\tGPU: %f    delta:%e", i, result_a[i], result_b[i], result_a[i]-result_b[i]);
            return -1;
        }
            
    }
    return 0;
}

int main(int argc, char const *argv[])
{
    //read matrix from file
    if (argc != 3)
    {
         printf("USAGE: ./matrix file_name");
         return 1;
    }

    int nvar = atoi(argv[2]);
    double* matrix = (double*)calloc((size_t)(nvar*(nvar+1)),(size_t)sizeof(double));

    if(read_file(argv[1], matrix, &nvar) != 0)
    {
        free(matrix);
        return -1;
    }
    printf("nvar: %d\n", nvar);
    
    printf("Initial matrix:\n\n");
    for(int i = 0; i< nvar; i++)
    {
        for (int j = 0; j <= nvar; j++)
            printf("%.1f\t",matrix[(nvar+1) *i + j]); 
        printf("\n"); 
    }
    //////////////////////////////////////////////////

	double result_a[nvar];
    double result_b[nvar];

	gauss_cpu(matrix, result_a, nvar);
    printf("\nThe values of the variables are as follows:\n");
    int i;
    for (i=0;i<nvar;i++)
        printf("%f\n",result_a[i]); 

    
    printf("\nGPU\n");

    if(read_file(argv[1], matrix, &nvar) != 0)
    {
        free(matrix);
        return -1;
    }

    gauss_gpu(matrix, result_b, nvar);
    printf("\nThe values of the variables are as follows:\n");
    //int i;
    for (i=0;i<nvar;i++)
        printf("%f\n",result_b[i]);   
    
    printf("\n***********************\n");
    if (verification(result_a, result_b, nvar) != 0)
    {
        /*printf("Result mismatch:\n");
        for(i = 0; i < nvar; i++)
        {
            printf("CPU: %f\tGPU: %f\n", result_a[i], result_b[i]);
        }*/
        free(matrix);
	    return -1;
    }

    printf("OK");
    free(matrix);
	return 0;
}