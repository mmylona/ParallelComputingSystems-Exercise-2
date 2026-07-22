#!/bin/bash
set -e
cd "$(dirname "$0")"

# Compile
mpicc -O3 -Wall mpi_polynomial_multiplication.c -o mpi_polynomial_multiplication

# Configuration
DEGREES=(50000 100000)
PROCESSES=(1 2 4 8)
REPEATS=4
OUT="./results.csv"

echo "Degree,Processes,AvgSerial,AvgSend,AvgCompute,AvgReceive,AvgTotal,AvgSpeedup,Correct" > $OUT

for n in "${DEGREES[@]}"; do
    for p in "${PROCESSES[@]}"; do
        sum_s=0; sum_snd=0; sum_c=0; sum_r=0; sum_t=0; sum_sp=0; status="YES"

        echo "Testing N=$n | Processes=$p"

        for ((r=1; r<=REPEATS; r++)); do
            output=$(mpirun --oversubscribe -np $p ./mpi_polynomial_multiplicationx -n $n)
            s=$(echo   "$output" | awk '{print $1}')   # serial time
            snd=$(echo "$output" | awk '{print $2}')   # send (Bcast) time
            c=$(echo   "$output" | awk '{print $3}')   # parallel compute time
            rc=$(echo  "$output" | awk '{print $4}')   # receive (Gatherv) time
            t=$(echo   "$output" | awk '{print $5}')   # total time
            sp=$(echo  "$output" | awk '{print $6}')   # speedup
            ok=$(echo  "$output" | awk '{print $7}')   # correctness

            sum_s=$(awk   -v a="$sum_s"   -v b="$s"   'BEGIN {print a + b}')
            sum_snd=$(awk -v a="$sum_snd" -v b="$snd" 'BEGIN {print a + b}')
            sum_c=$(awk   -v a="$sum_c"   -v b="$c"   'BEGIN {print a + b}')
            sum_r=$(awk   -v a="$sum_r"   -v b="$rc"  'BEGIN {print a + b}')
            sum_t=$(awk   -v a="$sum_t"   -v b="$t"   'BEGIN {print a + b}')
            sum_sp=$(awk  -v a="$sum_sp"  -v b="$sp"  'BEGIN {print a + b}')
            if [ "$ok" != "YES" ]; then status="NO"; fi
        done

        avg_s=$(awk   -v s="$sum_s"   -v r="$REPEATS" 'BEGIN {printf "%.6f", s / r}')
        avg_snd=$(awk -v s="$sum_snd" -v r="$REPEATS" 'BEGIN {printf "%.6f", s / r}')
        avg_c=$(awk   -v s="$sum_c"   -v r="$REPEATS" 'BEGIN {printf "%.6f", s / r}')
        avg_r=$(awk   -v s="$sum_r"   -v r="$REPEATS" 'BEGIN {printf "%.6f", s / r}')
        avg_t=$(awk   -v s="$sum_t"   -v r="$REPEATS" 'BEGIN {printf "%.6f", s / r}')
        avg_sp=$(awk  -v s="$sum_sp"  -v r="$REPEATS" 'BEGIN {printf "%.6f", s / r}')

        echo "$n,$p,$avg_s,$avg_snd,$avg_c,$avg_r,$avg_t,$avg_sp,$status" >> $OUT
    done
done

echo "Benchmarking complete. Data saved to $OUT"