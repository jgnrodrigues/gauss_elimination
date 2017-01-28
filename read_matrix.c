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
    if (argc != 2)
    {
         printf("USAGE: ./matrix file_name");
         return 1;
    }

    FILE *file;
    file=fopen(argv[1], "r");

    int nvar;
    if (!fscanf(file, "%d", &nvar))
    {
         printf("Can't read file");
         return 1;
    }

    printf("%d\n\n",nvar);

    if(nvar == 0)
    {
        return 0;
    }

    int i,j;

    /*matrix*/

    double* mat=malloc(sizeof(double)*nvar*(nvar+1));    

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