CC = mpicc
CFLAGS = -O3 -Wall

all: mpi_polynomial_multiplication

mpi_polynomial_multiplication: mpi_polynomial_multiplication.c ../utils.c ../utils.h
	$(CC) $(CFLAGS) -o mpi_polynomial_multiplication mpi_polynomial_multiplication.c ../utils.c

clean:
	rm -f mpi_polynomial_multiplication

.PHONY: all clean