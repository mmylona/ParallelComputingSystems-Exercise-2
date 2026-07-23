#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int    N;
    long  *data;
} DenseMatrix;

// CSR representation
typedef struct {
    int   N;
    int   nonZero;
    long  *nonZeroValues;
    int   *nonZeroColumnIdx;
    int   *rowPtr;
} CSRMatrix;

// Local slice per rank of a CSR matrix, received via MPI from rank 0
typedef struct {
    int    rows;
    int    nonZero;
    int   *rowPtr;
    int   *nonZeroColumnIdx;
    long  *nonZeroValues;
} LocalCSR;

static DenseMatrix *allocateDense(int N) {
    DenseMatrix *m = malloc(sizeof(DenseMatrix));
    m->N    = N;
    m->data = calloc((size_t)N * N, sizeof(long));
    return m;
}

static void freeDense(DenseMatrix *m) {
    free(m->data); 
    free(m);
}

static void freeCSR(CSRMatrix *m) {
    free(m->nonZeroValues); 
    free(m->nonZeroColumnIdx); 
    free(m->rowPtr); free(m);
}

static void freeLocalCSR(LocalCSR *m) {
    free(m->rowPtr); 
    free(m->nonZeroColumnIdx); 
    free(m->nonZeroValues);
}

static DenseMatrix *buildDense(int N, int sparsityPercent) {
    DenseMatrix *m = allocateDense(N);
    srand(42);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if ((rand() % 100) >= sparsityPercent)
                m->data[(size_t)i * N + j] = (rand() % 100) + 1;
    return m;
}

static long *buildVector(int N) {
    long *v = malloc(N * sizeof(long));
    srand(99);
    for (int i = 0; i < N; i++)
        v[i] = (rand() % 10) + 1;
    return v;
}

// run only by rank 0
static CSRMatrix *serialCsrBuild(DenseMatrix *d) {
    int N = d->N;
    CSRMatrix *csr = malloc(sizeof(CSRMatrix));
    csr->N      = N;
    csr->rowPtr = malloc((N + 1) * sizeof(int));

    csr->rowPtr[0] = 0;
    for (int i = 0; i < N; i++) {
        int cnt = 0;
        for (int j = 0; j < N; j++)
            if (d->data[(size_t)i * N + j] != 0) cnt++;
        csr->rowPtr[i + 1] = csr->rowPtr[i] + cnt;
    }
    csr->nonZero          = csr->rowPtr[N];
    csr->nonZeroValues    = malloc(csr->nonZero * sizeof(long));
    csr->nonZeroColumnIdx = malloc(csr->nonZero * sizeof(int));

    for (int i = 0; i < N; i++) {
        int pos = csr->rowPtr[i];
        for (int j = 0; j < N; j++) {
            long v = d->data[(size_t)i * N + j];
            if (v != 0) {
                csr->nonZeroValues[pos]    = v;
                csr->nonZeroColumnIdx[pos] = j;
                pos++;
            }
        }
    }
    return csr;
}

static void serialSpmvCsr(CSRMatrix *A, long *x, long *y) {
    for (int i = 0; i < A->N; i++) {
        long sum = 0;
        for (int k = A->rowPtr[i]; k < A->rowPtr[i + 1]; k++)
            sum += A->nonZeroValues[k] * x[A->nonZeroColumnIdx[k]];
        y[i] = sum;
    }
}

static int verify(long *a, long *b, int N) {
    for (int i = 0; i < N; i++)
        if (a[i] != b[i]) return 0;
    return 1;
}

static void splitRows(int N, int size, int rank, int *lo, int *hi) {
    int base = N / size, rem = N % size;
    if (rank < rem) { 
        *lo = rank * (base + 1); 
        *hi = *lo + base + 1; 
    }
    else {
        *lo = rem * (base + 1) + (rank - rem) * base; 
        *hi = *lo + base; 
    }
}

// Rank 0 sends every rank only the rows of the CSR matrix it needs
static LocalCSR distributeCSR(CSRMatrix *csr, int N, int rank, int size) {
    LocalCSR local;
    int lo, hi;
    splitRows(N, size, rank, &lo, &hi); 
    local.rows = hi - lo;
    local.rowPtr = malloc((local.rows + 1) * sizeof(int));

    if (rank == 0) {
        for (int r = 0; r < size; r++) {
            int l, h;
            splitRows(csr->N, size, r, &l, &h);
            int rrows = h - l;
            int rnnz  = csr->rowPtr[h] - csr->rowPtr[l];
            int *rp = malloc((rrows + 1) * sizeof(int));
            for (int i = 0; i <= rrows; i++) rp[i] = csr->rowPtr[l + i] - csr->rowPtr[l];

            if (r == 0) {
                local.nonZero = rnnz;
                memcpy(local.rowPtr, rp, (rrows + 1) * sizeof(int));
                local.nonZeroColumnIdx = malloc((rnnz > 0 ? rnnz : 1) * sizeof(int));
                local.nonZeroValues    = malloc((rnnz > 0 ? rnnz : 1) * sizeof(long));
                memcpy(local.nonZeroColumnIdx, &csr->nonZeroColumnIdx[csr->rowPtr[l]], rnnz * sizeof(int));
                memcpy(local.nonZeroValues,    &csr->nonZeroValues[csr->rowPtr[l]],    rnnz * sizeof(long));
            } else {
                MPI_Send(&rnnz, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
                MPI_Send(rp, rrows + 1, MPI_INT, r, 1, MPI_COMM_WORLD);
                MPI_Send(&csr->nonZeroColumnIdx[csr->rowPtr[l]], rnnz, MPI_INT, r, 2, MPI_COMM_WORLD);
                MPI_Send(&csr->nonZeroValues[csr->rowPtr[l]], rnnz, MPI_LONG, r, 3, MPI_COMM_WORLD);
            }
            free(rp);
        }
    } else {
        MPI_Recv(&local.nonZero, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local.rowPtr, local.rows + 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local.nonZeroColumnIdx = malloc((local.nonZero > 0 ? local.nonZero : 1) * sizeof(int));
        local.nonZeroValues    = malloc((local.nonZero > 0 ? local.nonZero : 1) * sizeof(long));
        MPI_Recv(local.nonZeroColumnIdx, local.nonZero, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local.nonZeroValues, local.nonZero, MPI_LONG, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    return local;
}

static void mpiSpmvCsrLocal(LocalCSR *local, long *x, long *yLocal) {
    for (int i = 0; i < local->rows; i++) {
        long sum = 0;
        for (int k = local->rowPtr[i]; k < local->rowPtr[i + 1]; k++)
            sum += local->nonZeroValues[k] * x[local->nonZeroColumnIdx[k]];
        yLocal[i] = sum;
    }
}

// row block owned by this rank instead of the whole N rows
static void mpiSpmvDenseLocal(long *localData, int localRows, int N, long *x, long *yLocal) {
    for (int i = 0; i < localRows; i++) {
        long sum = 0;
        for (int j = 0; j < N; j++)
            sum += localData[(size_t)i * N + j] * x[j];
        yLocal[i] = sum;
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0)
            fprintf(stderr, "Usage: mpirun -np <processes> %s <N> <sparsity_pct> <iterations>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    int N          = atoi(argv[1]);
    int sparsity   = atoi(argv[2]);
    int iterations = atoi(argv[3]);

    if (rank == 0)
        printf("N=%d sparsity=%d iterations=%d processes=%d\n", N, sparsity, iterations, size);

    DenseMatrix *D = NULL;
    long *inputVector = NULL;
    CSRMatrix *csrSerial = NULL;

    if (rank == 0) {
        D = buildDense(N, sparsity);
        inputVector = buildVector(N);
    } else {
        inputVector = malloc(N * sizeof(long)); 
    }

    //  CSR construction time
    double t_build = 0.0;
    if (rank == 0) {
        double t0 = MPI_Wtime();
        csrSerial = serialCsrBuild(D);
        t_build = MPI_Wtime() - t0;
        printf("CSR_BUILD %.6f nnz=%d\n", t_build, csrSerial->nonZero);
    }

    MPI_Bcast(inputVector, N, MPI_LONG, 0, MPI_COMM_WORLD);

    // distribution time
    MPI_Barrier(MPI_COMM_WORLD);
    double d0 = MPI_Wtime();
    LocalCSR local = distributeCSR(csrSerial, N, rank, size);
    MPI_Barrier(MPI_COMM_WORLD);
    double t_distribution = MPI_Wtime() - d0;
    if (rank == 0) printf("DISTRIBUTION %.6f\n", t_distribution);

    int *counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    for (int r = 0; r < size; r++) {
        int l, h; splitRows(N, size, r, &l, &h);
        counts[r] = h - l; displs[r] = l;
    }

    // parallel compute time with CSR representation
    long *x = malloc(N * sizeof(long));
    memcpy(x, inputVector, N * sizeof(long));
    long *yLocalCsr = malloc((local.rows > 0 ? local.rows : 1) * sizeof(long));

    MPI_Barrier(MPI_COMM_WORLD);
    double c0 = MPI_Wtime();
    for (int it = 0; it < iterations; it++) {
        mpiSpmvCsrLocal(&local, x, yLocalCsr);
        MPI_Allgatherv(yLocalCsr, local.rows, MPI_LONG, x, counts, displs, MPI_LONG, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t_spmv_csr = MPI_Wtime() - c0;
    if (rank == 0) printf("SPMV_CSR_PARALLEL %.6f\n", t_spmv_csr);

    double t_total_csr = t_build + t_distribution + t_spmv_csr;
    if (rank == 0) printf("TOTAL_CSR %.6f\n", t_total_csr);

    long *ySerial = NULL, *xSerial = NULL;
    double t_spmv_serial = 0.0;
    if (rank == 0) {
        xSerial = malloc(N * sizeof(long));
        ySerial = malloc(N * sizeof(long));
        memcpy(xSerial, inputVector, N * sizeof(long));
        double s0 = MPI_Wtime();
        for (int it = 0; it < iterations; it++) {
            serialSpmvCsr(csrSerial, xSerial, ySerial);
            memcpy(xSerial, ySerial, N * sizeof(long));
        }
        t_spmv_serial = MPI_Wtime() - s0;
        printf("SPMV_CSR_SERIAL %.6f\n", t_spmv_serial);

        int ok = verify(xSerial, x, N);
        printf("VERIFICATION %s\n", ok ? "OK" : "FAIL");
    }

    free(counts); free(displs);
    free(x); free(yLocalCsr);
    freeLocalCSR(&local);

    // dense representation
    int myLo, myHi;
    splitRows(N, size, rank, &myLo, &myHi);
    int myRows = myHi - myLo;
    long *localDense = malloc((size_t)(myRows > 0 ? myRows : 1) * N * sizeof(long));

    MPI_Barrier(MPI_COMM_WORLD);
    double e0 = MPI_Wtime();
    if (rank == 0) {
        for (int r = 0; r < size; r++) {
            int l, h; splitRows(N, size, r, &l, &h);
            int rrows = h - l;
            if (r == 0)
                memcpy(localDense, &D->data[(size_t)l * N], (size_t)rrows * N * sizeof(long));
            else
                MPI_Send(&D->data[(size_t)l * N], rrows * N, MPI_LONG, r, 4, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(localDense, myRows * N, MPI_LONG, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    long *xDense = malloc(N * sizeof(long));
    memcpy(xDense, inputVector, N * sizeof(long));
    long *yLocalDense = malloc((myRows > 0 ? myRows : 1) * sizeof(long));
    counts = malloc(size * sizeof(int));
    displs = malloc(size * sizeof(int));
    for (int r = 0; r < size; r++) {
        int l, h; splitRows(N, size, r, &l, &h);
        counts[r] = h - l; displs[r] = l;
    }

    for (int it = 0; it < iterations; it++) {
        mpiSpmvDenseLocal(localDense, myRows, N, xDense, yLocalDense);
        MPI_Allgatherv(yLocalDense, myRows, MPI_LONG, xDense, counts, displs, MPI_LONG, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t_spmv_dense = MPI_Wtime() - e0;
    if (rank == 0) printf("SPMV_DENSE_PARALLEL %.6f\n", t_spmv_dense);

    free(localDense); free(xDense); free(yLocalDense); free(counts); free(displs);
    free(inputVector);
    if (rank == 0) { freeDense(D); freeCSR(csrSerial); free(xSerial); free(ySerial); }

    MPI_Finalize();
    return 0;
}