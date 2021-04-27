#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <math.h>

//DEFINE TYPESIZE as 4 because using unsigned int, maybe change to unsigned long
#define TYPESIZE 4

/* tell compiler to use register for faster calculation */
unsigned int gcd(register unsigned int u, register unsigned int v)
{
    if (u == 0)
        return v;
    if (v == 0)
        return u;
    if (u == v)
        return u;
    int shift;
    shift = __builtin_ctz(u | v);
    u >>= __builtin_ctz(u);
    do
    {
        v >>= __builtin_ctz(v);
        if (u > v)
        {
            register unsigned int t = v;
            v = u;
            u = t;
        }
        v = v - u;
    } while (v != 0);
    return u << shift;
}

int floor_log2(unsigned int x){
    if (x==0){
        return NULL;
    }
    return(31 - __builtin_clz(x));
}

unsigned int main(int argc, char *argv[])
{
    int myrank, npes;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    unsigned int size = 0;
    if (argc != 1)
    {
        size = argc - 1;
    }
    else
    {
        if (myrank == 0)
        {
            printf("\nNo Arguments");
        }
        return 0;
    }

    unsigned int rsize = (size - (size % npes)) / npes + 1;
    if (size % npes == 0)
    {
        rsize = size / npes;
    }
    unsigned int r[npes * rsize];

    register unsigned int i = 0;
    if (argc != 1)
    {
        while (--argc > 1)
        {
            r[i++] = (unsigned int)strtoul(*++argv, argv, 10);
        }
        r[i++] = (unsigned int)strtoul(*++argv, argv, 10);
    }
    for (i = size; i < (rsize * npes); ++i)
    {
        r[i] = 0;
    }

    unsigned int *jump = rsize * myrank * TYPESIZE + (void *)&r;
    /* Syncronize */
    MPI_Barrier(MPI_COMM_WORLD);

    for (i = rsize - 2; i < rsize; --i)
    {
        *(i + jump) = gcd(*(i + jump), *((i + 1) + jump));
    }
    for (i = 0; i < log2(npes); ++i)
    {
        int partner = myrank ^ (int)pow(2, i);
        if (partner < myrank && partner > -1)
        {
            MPI_Send(jump, 1, MPI_UNSIGNED, partner, 0, MPI_COMM_WORLD);
        }
        if (partner > myrank && partner < npes)
        {
            MPI_Recv((rsize + jump), 1, MPI_UNSIGNED, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            *jump = gcd(*jump, *(rsize + jump));
        }
    }

    if (myrank == 0)
    {
        printf("\nPairwise GCD: %d\n", r[0]);
    }
    MPI_Finalize();
    return 0;
}