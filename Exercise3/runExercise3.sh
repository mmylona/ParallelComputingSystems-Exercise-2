#!/bin/bash
cd "$(dirname "$0")"

# Compile
gcc -O2 -Wall -mavx2 multiplicationOfPolynomialsSIMD.c ../utils.c -o multiplicationOfPolynomialsSIMD

# Configuration
# NOTE: this is an O(n^2) algorithm, so very large n gets slow fast
# (n=10^6 alone is ~10^12 scalar multiply-adds, i.e. minutes per run).
# Add 1000000 here yourself if you have the time/patience for it.
DEGREES=(10000 50000 100000 200000)
REPEATS=4
OUT="./results.csv"

echo "Degree,AvgCreation,AvgScalar,AvgSIMD,AvgSpeedup,Correct" > $OUT

for n in "${DEGREES[@]}"; do
    sum_cr=0; sum_s=0; sum_v=0; sum_sp=0; status="YES"

    echo "Testing N=$n"

    for ((r=1; r<=REPEATS; r++)); do
        output=$(./multiplicationOfPolynomialsSIMD -n $n)
        cr=$(echo "$output" | awk '{print $1}')
        s=$(echo  "$output" | awk '{print $2}')
        v=$(echo  "$output" | awk '{print $3}')
        sp=$(echo "$output" | awk '{print $4}')
        ok=$(echo "$output" | awk '{print $5}')

        sum_cr=$(awk -v a="$sum_cr" -v b="$cr" 'BEGIN {print a + b}')
        sum_s=$(awk  -v a="$sum_s"  -v b="$s"  'BEGIN {print a + b}')
        sum_v=$(awk  -v a="$sum_v"  -v b="$v"  'BEGIN {print a + b}')
        sum_sp=$(awk -v a="$sum_sp" -v b="$sp" 'BEGIN {print a + b}')
        if [ "$ok" != "YES" ]; then status="NO"; fi
    done

    avg_cr=$(awk -v s="$sum_cr" -v r="$REPEATS" 'BEGIN {printf "%.6f", s / r}')
    avg_s=$(awk  -v s="$sum_s"  -v r="$REPEATS" 'BEGIN {printf "%.6f", s / r}')
    avg_v=$(awk  -v s="$sum_v"  -v r="$REPEATS" 'BEGIN {printf "%.6f", s / r}')
    avg_sp=$(awk -v s="$sum_sp" -v r="$REPEATS" 'BEGIN {printf "%.6f", s / r}')

    echo "$n,$avg_cr,$avg_s,$avg_v,$avg_sp,$status" >> $OUT
done

echo "Benchmarking complete. Data saved to $OUT"