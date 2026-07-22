#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    long long *coef;
    int degree;
} Polynomial;

typedef struct {
    const Polynomial *A, *B;
    Polynomial *C;
    int kStart, kEnd;
} RangeArgs;

static double now_sec(void) { return MPI_Wtime(); }

// Generate a random degree-n polynomial 
static Polynomial createRandomPolynomial(int n, unsigned int *seed) {
    Polynomial p;
    p.degree = n;
    p.coef = (long long *)malloc((n + 1) * sizeof(long long));
    if (!p.coef) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i <= n; i++) {
        long long v = rand_r(seed) % 100 + 1;
        if (rand_r(seed) % 2 == 0)
            v = -v;
        p.coef[i] = v;
    }
    return p;
}

// Allocate an degree-n polynomial with no values set yet
static Polynomial allocateEmptyPolynomial(int n) {
    Polynomial p;
    p.degree = n;
    p.coef = (long long *)malloc((n + 1) * sizeof(long long));
    if (!p.coef) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    return p;
}

// Allocate a zero polynomial of degree a+b 
static Polynomial allocateZeroResultPolynomial(int degreeA, int degreeB) {
    Polynomial r;
    r.degree = degreeA + degreeB;
    r.coef = (long long *)calloc(degreeA + degreeB + 1, sizeof(long long));
    if (!r.coef) {
        perror("calloc");
        exit(EXIT_FAILURE);
    }
    return r;
}

static void serialMultiply(const Polynomial *A, const Polynomial *B, Polynomial *C) {
    int degreeA = A->degree;
    int degreeB = B->degree;
    for (int i = 0; i <= degreeA; i++)
    for (int j = 0; j <= degreeB; j++)
        C->coef[i + j] += (long long)A->coef[i] * B->coef[j];
}

// Is called directly by each MPI process for its own range
static void computeRange(RangeArgs *a) {
    int degreeA = a->A->degree;
    int degreeB = a->B->degree;

    for (int k = a->kStart; k <= a->kEnd; k++) {
        long long sum = 0;
        int start = (k - degreeB > 0) ? k - degreeB : 0;
        int end = (k < degreeA) ? k : degreeA;
        for (int i = start; i <= end; i++)
            sum += a->A->coef[i] * a->B->coef[k - i];
        a->C->coef[k] = sum;
    }
}

static void splitWork(int total, int size, int rank, int *kStart, int *kEnd) {
    int workPerRank = total / size;
    int remainder = total % size;
    int start = 0;
    for (int r = 0; r < rank; r++)
    start += workPerRank + (r < remainder ? 1 : 0);
    int end = start + workPerRank - 1 + (rank < remainder ? 1 : 0);
    *kStart = start;
    *kEnd = end;
}

// Runs the distributed MPI multiplication of A*B 
static void mpiMultiply(const Polynomial *A, const Polynomial *B, Polynomial *C, int rank, int size, double *sendTime, double *computeTime, double *recvTime) {
    // number of output coefficients
    int total = A->degree + B->degree + 1; 

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = now_sec();

    // distribute the necessary data
    MPI_Bcast(A->coef, A->degree + 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(B->coef, B->degree + 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = now_sec();

    // parallel computation
    int kStart, kEnd;
    splitWork(total, size, rank, &kStart, &kEnd);
    RangeArgs range = {A, B, C, kStart, kEnd};
    computeRange(&range);

    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = now_sec();

    // gather each ranks slice of C back into rank 0
    int *counts = NULL, *displs = NULL;
    if (rank == 0) {
        counts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        for (int r = 0; r < size; r++) {
            int s, e;
            splitWork(total, size, r, &s, &e);
            counts[r] = e - s + 1;
            displs[r] = s;
        }
    }
    int localCount = kEnd - kStart + 1;
    MPI_Gatherv(&C->coef[kStart], localCount, MPI_LONG_LONG, C->coef, counts,
                displs, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t3 = now_sec();

    *sendTime = t1 - t0;
    *computeTime = t2 - t1;
    *recvTime = t3 - t2;

    free(counts);
    free(displs);
}

// Verify two result polynomials are identical
static int isEqual(const Polynomial *P, const Polynomial *Q) {
    if (P->degree != Q->degree)
        return 0;
    for (int i = 0; i <= P->degree; i++)
        if (P->coef[i] != Q->coef[i])
            return 0;
    return 1;
}

static void freePolynomial(Polynomial *p) {
    free(p->coef);
    p->coef = NULL;
    p->degree = -1;
}

static void usage(const char *prog) {
    fprintf(stderr, "Error: Invalid Arguments\n");
    fprintf(stderr, "Usage: mpirun -np <processes> %s -n <degree>\n", prog);
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = -1; // degree of both polynomials

    if (argc < 3) {
        if (rank == 0) 
            usage(argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc)
            n = atoi(argv[++i]);
        else {
            if (rank == 0) 
                fprintf(stderr, "Error: Unknown flag '%s'\n", argv[i]);
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    if (n < 0) {
        if (rank == 0) 
            fprintf(stderr, "Error: Missing or invalid parameters.\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // initialisation
    unsigned int seed = (unsigned int)time(NULL);
    Polynomial A, B;
    if (rank == 0) {
        A = createRandomPolynomial(n, &seed);
        B = createRandomPolynomial(n, &seed);
    } else {
        A = allocateEmptyPolynomial(n);
        B = allocateEmptyPolynomial(n);
    }

    Polynomial C_parallel = allocateZeroResultPolynomial(A.degree, B.degree);
    Polynomial C_serial = {NULL, -1};
    if (rank == 0)
        C_serial = allocateZeroResultPolynomial(A.degree, B.degree);

    // serial multiplication
    double serialTime = 0.0;
    if (rank == 0) {
        double s0 = now_sec();
        serialMultiply(&A, &B, &C_serial);
        serialTime = now_sec() - s0;
    }

    // parallel MPI multiplication
    double sendTime, computeTime, recvTime;
    mpiMultiply(&A, &B, &C_parallel, rank, size, &sendTime, &computeTime, &recvTime);
    double totalTime = sendTime + computeTime + recvTime;

    if (rank == 0) {
        int ok = isEqual(&C_serial, &C_parallel);
        double speedup = (totalTime > 0) ? (serialTime / totalTime) : 0;

        printf("%.6f %.6f %.6f %.6f %.6f %.2f %s\n", serialTime, sendTime, computeTime, recvTime, totalTime, speedup, ok ? "YES" : "NO");
    }

    freePolynomial(&A);
    freePolynomial(&B);
    freePolynomial(&C_parallel);
    if (rank == 0) 
        freePolynomial(&C_serial);

    MPI_Finalize();
    return EXIT_SUCCESS;
}