CC      = mpicc
CFLAGS  = -O2 -Wall
LDFLAGS =

all: mpi_spmv

spmv_mpi: mpi_spmv.c
	$(CC) $(CFLAGS) -o $@ mpi_spmv.c $(LDFLAGS)

clean:
	rm -f mpi_spmv