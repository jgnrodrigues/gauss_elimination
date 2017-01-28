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
            
            if(k > i) //compute if the idx's line is below of i's line
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
    printf("\ngaussOnGPU<<<%d, %d>>>\n", grid.x * grid.y, block.x * block.y);  

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

int main(int argc, char const *argv[])
{
	int nvar = 4;
    double* matrix = (double*)calloc((size_t)(nvar*(nvar+1)),(size_t)sizeof(double));

    nvar = 4 + 1;
    matrix[nvar*0+0] =  4.0;
    matrix[nvar*0+1] = -3.0;
    matrix[nvar*0+2] =  0.0;
    matrix[nvar*0+3] =  1.0;
    matrix[nvar*0+4] = -7.0;

    matrix[nvar*1+0] =  2.0;
    matrix[nvar*1+1] =  2.0;
    matrix[nvar*1+2] =  3.0;
    matrix[nvar*1+3] =  2.0;
    matrix[nvar*1+4] = -2.0;

    matrix[nvar*2+0] =  6.0;
    matrix[nvar*2+1] =  1.0;
    matrix[nvar*2+2] = -6.0;
    matrix[nvar*2+3] = -5.0;
    matrix[nvar*2+4] =  6.0;

    matrix[nvar*3+0] =  0.0;
    matrix[nvar*3+1] =  2.0;
    matrix[nvar*3+2] =  0.0;
    matrix[nvar*3+3] =  1.0;
    matrix[nvar*3+4] =  0.0;
    
    nvar = 4;
	double result[nvar];
	/*gauss_cpu(matrix, result, nvar);
    printf("\nThe values of the variables are as follows:\n");
    int i;
    for (i=0;i<nvar;i++)
        printf("%f\n",result[i]);  */
    printf("\nGPU\n");
    
    nvar = 4 + 1;
    matrix[nvar*0+0] =  4.0;
    matrix[nvar*0+1] = -3.0;
    matrix[nvar*0+2] =  0.0;
    matrix[nvar*0+3] =  1.0;
    matrix[nvar*0+4] = -7.0;

    matrix[nvar*1+0] =  2.0;
    matrix[nvar*1+1] =  2.0;
    matrix[nvar*1+2] =  3.0;
    matrix[nvar*1+3] =  2.0;
    matrix[nvar*1+4] = -2.0;

    matrix[nvar*2+0] =  6.0;
    matrix[nvar*2+1] =  1.0;
    matrix[nvar*2+2] = -6.0;
    matrix[nvar*2+3] = -5.0;
    matrix[nvar*2+4] =  6.0;

    matrix[nvar*3+0] =  0.0;
    matrix[nvar*3+1] =  2.0;
    matrix[nvar*3+2] =  0.0;
    matrix[nvar*3+3] =  1.0;
    matrix[nvar*3+4] =  0.0;

    nvar = 4;
    gauss_gpu(matrix, result, nvar);
    printf("\nThe values of the variables are as follows:\n");
    int i;
    for (i=0;i<nvar;i++)
        printf("%f\n",result[i]);

    free(matrix);
	return 0;
}