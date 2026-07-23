CC      = gcc
CFLAGS  = -O2 -Wall -mavx2
LDFLAGS =

all: multiplicationOfPolynomialsSIMD

multiplicationOfPolynomialsSIMD: multiplicationOfPolynomialsSIMD.c ../utils.c
	$(CC) $(CFLAGS) -o $@ multiplicationOfPolynomialsSIMD.c ../utils.c $(LDFLAGS)

clean:
	rm -f multiplicationOfPolynomialsSIMD