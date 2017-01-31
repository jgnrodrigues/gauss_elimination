/////////////////////////////////////////////////////////////////////
//  RESOLVING SYSTEM OF LINEAR EQUATIONS USING GAUSSIAN ELIMINATION
//  CUDA
//
//  João Rodrigues
//  MEC: 71771
//
//  
/////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "common/common.h"
#include <cuda_runtime.h>

//cpu

/////////////////////////////////
// 0 -> independent system
// 1 -> dependent system
// 2 -> inconsistent system
/////////////////////////////////

int gauss_cpu(double matrix[], double result[], int n)
{
	int i,j,k;
    int nvar = n + 1;
    double tmp;
	
	for (i = 0; i < n - 1; i++)
	{                    
        /*
        for (k=i+1;k<n;k++)     //Pivotisation biggest
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
        */

        //Pivotisation non zero
        if (abs(matrix[nvar * i + i]) < 1e-13)                 
            for (k=i+1;k<n;k++)
            {
                if (abs(matrix[nvar * i + i]) < 1e-13)
                {
                    for (j=0;j<=n;j++)
                    {
                        double temp=matrix[nvar*i+j];
                        matrix[nvar*i+j]=matrix[nvar*k+j];
                        matrix[nvar*k+j]=temp;
                    }
                    break;
                }
            }
        //loop to perform the gauss elimination
        /*
            i -> diagonal row
            k -> rows below
            j -> collumn
        */
        for (k=i+1;k<n;k++)
		{
			double ratio=matrix[nvar*k+i]/matrix[nvar*i+i]; 
			for (j=0;j<=n;j++)
				matrix[nvar*k+j]=matrix[nvar*k+j]-ratio*matrix[nvar*i+j];    //make the elements below the pivot elements equal to zero or elimnate the variables
                if(abs(matrix[nvar*k+j]) < 1e-13)
                    matrix[nvar*k+j] = 0.0;
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

    // type verification
    int zero, type = 0;
    for (i=n-1;i>=0;i--)
    {
        zero = 1;
        for (j=0;j<n;j++)
            if (abs(matrix[nvar*i+j]) > 1e-13)
            {
                zero = 0;
                break;
            }
        if (zero == 1)
        {
            if(matrix[nvar*i+j] < 1e-13)
                type = 1;
            else 
                return 2;
        }
    }
    if (type == 1)
        return 1;

    ////////////////////////////////////////

    for (i = n - 1; i >= 0; i--) {
		tmp = matrix[nvar*i+n];
		for (j = n - 1; j > i; j--)
			tmp -= result[j] * matrix[nvar * i + j];
        result[i] = tmp / matrix[nvar * i + i];
	}
    return 0;
}

//gauss elimination   CUDA kernel
__global__ void ratioOnGPU(double matrix[], double ratio[], const int n, const int i)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * gridDim.x * blockDim.x + ix;
    int line_size = n + 1;

    if (idx < (gridDim.x * gridDim.y * blockDim.x * blockDim.y)&& idx <= n - i)
    {
        ratio[idx] = matrix[line_size*(idx+1)+i]/matrix[i];
    }
}

__global__ void gaussOnGPU(double matrix[], double ratio[], const int n, const int i)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * gridDim.x * blockDim.x + ix;
    int line_size = n + 1;

    int k = idx / line_size; //line of matrix wich idx belongs
    k += 1;
    int j = idx % line_size; //collumn of matrix wich idx belongs

    if (idx < (gridDim.x * gridDim.y * blockDim.x * blockDim.y)&& k < n - i && j <= n)
    {
        matrix[line_size*k+j] = matrix[line_size*k+j]-ratio[k-1]*matrix[j];       //make the elements below the pivot elements equal to zero or elimnate the variables
        if(abs(matrix[line_size*k+j]) < 1e-13)
            matrix[line_size*k+j] = 0.0;
    }
}




//GPU
/////////////////////////////////
// 0 -> independent system
// 1 -> dependent system
// 2 -> inconsistent system
/////////////////////////////////
int gauss_gpu(double matrix[], double result[], int n)
{
    //setup device
    int dev = 0;
    double *dev_matrix, *dev_ratio;
    double tmp;
    cudaDeviceProp deviceProp;

    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    size_t matrix_size = sizeof(double) * n * (n + 1);
    size_t result_size = sizeof(double) * n;

    if(matrix_size + result_size > deviceProp.totalGlobalMem)
    {
        fprintf(stderr,"Not enough memory in the cuda device\n");
        exit(1);
    }

    int i,j,k;
    int nvar = n + 1;
    double* temp = (double*)calloc((size_t)(nvar+1), (size_t)(sizeof(double)));
	
	for (i = 0; i < n - 1; i++)
	{   
        //Pivotisation
        if (abs(matrix[nvar * i + i]) < 1e-13)                 
            for (k=i+1;k<n;k++)
            {
                if (abs(matrix[nvar * i + i]) < 1e-13)
                {
                    mempcpy(temp, matrix+nvar*i, sizeof(double)*nvar);
                    mempcpy(matrix+nvar*i, matrix+nvar*k, sizeof(double)*nvar);
                    mempcpy(matrix+nvar*k, temp, sizeof(double)*nvar);
                    break;
                }
            }

        //gauss elimination
        /*
            i -> diagonal row
            k -> rows below
            j -> collumn
        */

         //device memory allocation
        CHECK(cudaMalloc((void **)&dev_matrix, (matrix_size - i * nvar)));
        CHECK(cudaMalloc((void **)&dev_ratio, (n - (i + 1))));

        // transfer data from host to device
        CHECK(cudaMemcpy(dev_matrix, (matrix + i * nvar), (matrix_size - i * nvar), cudaMemcpyHostToDevice));

        // device configurations
        int block_x = 1;
        int block_y = n - (i + 1); //only elements that need to be calculated

        int grid_x = 1;
        int grid_y = 1;

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

        dim3 block (block_x, block_y);
        dim3 grid  (grid_x, grid_y);

        //kernel execution
        printf("\n i:%d ratioOnGPU<<<%d, %d>>>: block: %d x %d\n", i, grid.x * grid.y, block.x * block.y, block.x, block.y);
        ratioOnGPU<<<grid, block>>>(dev_matrix, dev_ratio, n, i);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        //device configurations
        block.x = n + 1;
        block.y = n - (i + 1); //only elements that need to be calculated

        grid.x = 1;
        grid.y = 1;

        //block size verification
        if(block.x > deviceProp.maxThreadsDim[0] || block.y > deviceProp.maxThreadsDim[1] || block.x * block.y > deviceProp.maxThreadsPerBlock)
        {
            fprintf(stderr,"Block too big\n");
            exit(1);
        }

        //grid size verification
        if(grid.x > deviceProp.maxGridSize[0] || grid.y > deviceProp.maxGridSize[1])
        {
            fprintf(stderr,"Grid too big\n");
            exit(1);
        }

        //kernel execution
        printf("\n i:%d gaussOnGPU<<<%d, %d>>>: block: %d x %d\n", i, grid.x * grid.y, block.x * block.y, block.x, block.y);
        gaussOnGPU<<<grid, block>>>(dev_matrix, dev_ratio, n, i);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
       

        // transfer data from device to host
        CHECK(cudaMemcpy((matrix + i * nvar), dev_matrix, (matrix_size - i * nvar), cudaMemcpyDeviceToHost));

        //free device memory
        CHECK(cudaFree(dev_matrix));
        CHECK(cudaFree(dev_ratio));

        printf("\n\n i:%d  The matrix in %d iteration:\n", i, i);
        int a, b;
        for (a=0;a<n;a++)            //print the new matrix
        {
            for (b=0;b<=n;b++)
                printf("%f\t",matrix[nvar*a+b]);
            printf("\n");
        }  
	}
    free(temp);

    /////////////////////////////
     printf("\n\nThe matrix after gauss-elimination is as follows:\n");
     for (i=0;i<n;i++)            //print the new matrix
    {
        for (j=0;j<=n;j++)
            printf("%f\t",matrix[nvar*i+j]);
        printf("\n");
    }    
    ////////////////////////////

    // type verification
    int zero, type = 0;
    for (i=n-1;i>=0;i--)
    {
        zero = 1;
        for (j=0;j<n;j++)
            if (abs(matrix[nvar*i+j]) > 1e-13)
            {
                zero = 0;
                break;
            }
        if (zero == 1)
        {
            if(matrix[nvar*i+j] < 1e-13)
                type = 1;
            else 
                return 2;
        }
    }
    if (type == 1)
        return 1;

    ///////////////////////////////////

	for (i = n - 1; i >= 0; i--) {  //back substitution
		tmp = matrix[nvar*i+n];
		for (j = n - 1; j > i; j--)
			tmp -= result[j] * matrix[nvar * i + j];
        result[i] = tmp / matrix[nvar * i + i];
	}
    return 0;

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

    int i;

    /*matrix*/

    for(i = 0; i < nvar * (nvar+1); i++)
    {
        //Use lf format specifier, %c is for character
        if (!fscanf(file, "%lf", matrix +i )) 
            break;
    }
    fclose(file);  

    *n = nvar;
    return 0;
}

int verification(double* result_a, double* result_b, int n)
{
    int i;
    int error = 0;
    for(i = 0; i < n; i++)
    {
        if(abs(result_a[i] - result_b[i]) > 1e-13)
        {   
            if (error == 0)
                printf("Result mismatch:\n");
            printf("i: %d\tCPU: %f\tGPU: %f    delta:%e\n", i, result_a[i], result_b[i], abs(result_a[i]-result_b[i]));
            error = -1;
        }            
    }
    return error;
}

int main(int argc, char const *argv[])
{
    int i,j;
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
    for(i = 0; i< nvar; i++)
    {
        for (j = 0; j <= nvar; j++)
            printf("%.1f\t",matrix[(nvar+1) *i + j]); 
        printf("\n"); 
    }
    //////////////////////////////////////////////////

	double result_a[nvar];
    double result_b[nvar];

    int type_a = 0, type_b = 0;

	type_a = gauss_cpu(matrix, result_a, nvar);
    if (type_a == 0)
    {
        printf("\nThe values of the variables are as follows:\n");
        for (i=0;i<nvar;i++)
            printf("%f\n",result_a[i]); 
    }
    else if(type_a == 1)
    {
        printf("\nThe system is dependent\n");
    }
    else if(type_a == 2)
    {
        printf("\nThe system is inconsistent\n");
    }
    
    printf("\nGPU\n");

    if(read_file(argv[1], matrix, &nvar) != 0)
    {
        free(matrix);
        return -1;
    }

    gauss_gpu(matrix, result_b, nvar);
    if (type_b == 0)
    {
        printf("\nThe values of the variables are as follows:\n");
        for (i=0;i<nvar;i++)
            printf("%f\n",result_b[i]);   
    }
    else if(type_b == 1)
    {
        printf("\nThe system is dependent\n");
    }
    else if(type_b == 2)
    {
        printf("\nThe system is inconsistent\n");
    }

    printf("\n***********************\n");
    if (type_a == 0 && type_b == 0)
    {
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
    }

    printf("OK");
    free(matrix);
	return 0;
}