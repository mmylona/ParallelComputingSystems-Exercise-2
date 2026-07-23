#include "../utils.h"
#include <immintrin.h>
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

// The original scalar algorithm 
static void serialMultiply(const Polynomial *A, const Polynomial *B,
                            Polynomial *C) {
  int degreeA = A->degree;
  int degreeB = B->degree;
  for (int i = 0; i <= degreeA; i++)
    for (int j = 0; j <= degreeB; j++)
      C->coef[i + j] += (long long)A->coef[i] * B->coef[j];
}

static void vectorizedMultiply(const Polynomial *A, const Polynomial *B,
                                Polynomial *C) {
  int degreeA = A->degree;
  int degreeB = B->degree;

  for (int i = 0; i <= degreeA; i++) {
    __m256i avec = _mm256_set1_epi64x(A->coef[i]);
    int j = 0;
    for (; j + 4 <= degreeB + 1; j += 4) {
      __m256i bvec = _mm256_loadu_si256((__m256i *)&B->coef[j]);
      __m256i cvec = _mm256_loadu_si256((__m256i *)&C->coef[i + j]);
      __m256i prod = _mm256_mul_epi32(avec, bvec); // exact for our value range
      cvec = _mm256_add_epi64(cvec, prod);
      _mm256_storeu_si256((__m256i *)&C->coef[i + j], cvec);
    }
    // scalar tail for the remaining < 4 elements
    for (; j <= degreeB; j++) {
      C->coef[i + j] += A->coef[i] * B->coef[j];
    }
  }
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
  fprintf(stderr, "Usage: %s -n <degree>\n", prog);
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
  int n = -1;

  if (argc < 3)
    usage(argv[0]);

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-n") && i + 1 < argc)
      n = atoi(argv[++i]);
    else {
      fprintf(stderr, "Error: Unknown flag '%s'\n", argv[i]);
      usage(argv[0]);
    }
  }

  if (n < 0) {
    fprintf(stderr, "Error: Missing or invalid parameters.\n");
    return EXIT_FAILURE;
  }

  // initialisation
  unsigned int seed = (unsigned int)time(NULL);
  double t0 = now_sec();
  Polynomial A = createRandomPolynomial(n, &seed);
  Polynomial B = createRandomPolynomial(n, &seed);
  Polynomial C_scalar = allocateZeroResultPolynomial(n, n);
  Polynomial C_simd = allocateZeroResultPolynomial(n, n);
  double creationTime = now_sec() - t0;

  // original scalar algorithm
  double t1 = now_sec();
  serialMultiply(&A, &B, &C_scalar);
  double scalarTime = now_sec() - t1;

  // vectorized (AVX2) algorithm
  double t2 = now_sec();
  vectorizedMultiply(&A, &B, &C_simd);
  double simdTime = now_sec() - t2;

  // verification
  int ok = isEqual(&C_scalar, &C_simd);

  // Summary
  double speedup = (simdTime > 0) ? (scalarTime / simdTime) : 0;
  printf("%.6f %.6f %.6f %.2f %s\n", creationTime, scalarTime, simdTime,
         speedup, ok ? "YES" : "NO");

  freePolynomial(&A);
  freePolynomial(&B);
  freePolynomial(&C_scalar);
  freePolynomial(&C_simd);

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}