////////////////////////////////////////////////////////
//  READ MATRIX FROM FILE
//  João Rodrigues
//  MEC: 71771
////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char const *argv[])
{
    if (argc != 3)
    {
         printf("USAGE: ./matrix system_nvar file_name");
         return 1;
    }

    int nvar = atoi(argv[1]);

    if(nvar == 0)
    {
        return 0;
    }


    int i,j;

    /*matrix*/

    double* mat=malloc(sizeof(double)*nvar*(nvar+1));

    FILE *file;
    file=fopen(argv[2], "r");

    for(i = 0; i < nvar * (nvar+1); i++)
    {
        //Use lf format specifier, %c is for character
        if (!fscanf(file, "%lf", &mat[i])) 
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

    for(int i = 0; i< nvar; i++)
    {
        for (int j = 0; j <= nvar; j++)
            printf("%.1f\t",mat[(nvar+1) *i + j]); 
        printf("\n"); 
    }


    return 0;
}