/////////////////////////////////////////////////////////////////////
//  RESOLVING SYSTEM OF LINEAR EQUATIONS USING GAUSSIAN ELIMINATION
//  CPU
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
	gauss_cpu(matrix, result, nvar);
    printf("\nThe values of the variables are as follows:\n");
    int i;
    for (i=0;i<nvar;i++)
        printf("%f\n",result[i]);           
	return 0;
}