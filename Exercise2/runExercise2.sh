#!/bin/bash
set -e
export LC_ALL=C

cd "$(dirname "$0")"
make -s

REPS=4
OUTDIR="results"
mkdir -p "$OUTDIR"

echo "=== Exercise 2.2: SpMV CSR (MPI) ==="
echo "REPS=$REPS"
echo ""

run_avg() {
    local N=$1 SP=$2 IT=$3 P=$4
    local sum_build=0 sum_dist=0 sum_cs=0 sum_cp=0 sum_total=0 sum_dp=0

    for r in $(seq 1 "$REPS"); do
        out=$(mpirun --oversubscribe -np "$P" ./mpi_spmv "$N" "$SP" "$IT")
        t_build=$(echo "$out" | awk '/^CSR_BUILD/{print $2}')
        t_dist=$(echo  "$out" | awk '/^DISTRIBUTION/{print $2}')
        t_cs=$(echo    "$out" | awk '/^SPMV_CSR_SERIAL/{print $2}')
        t_cp=$(echo    "$out" | awk '/^SPMV_CSR_PARALLEL/{print $2}')
        t_total=$(echo "$out" | awk '/^TOTAL_CSR/{print $2}')
        t_dp=$(echo    "$out" | awk '/^SPMV_DENSE_PARALLEL/{print $2}')
        sum_build=$(echo "$sum_build + $t_build" | bc -l)
        sum_dist=$(echo  "$sum_dist + $t_dist"   | bc -l)
        sum_cs=$(echo    "$sum_cs + $t_cs"       | bc -l)
        sum_cp=$(echo    "$sum_cp + $t_cp"       | bc -l)
        sum_total=$(echo "$sum_total + $t_total" | bc -l)
        sum_dp=$(echo    "$sum_dp + $t_dp"       | bc -l)
    done

    echo \
        "$(echo "scale=6; $sum_build/$REPS" | bc -l)" \
        "$(echo "scale=6; $sum_dist/$REPS"  | bc -l)" \
        "$(echo "scale=6; $sum_cs/$REPS"    | bc -l)" \
        "$(echo "scale=6; $sum_cp/$REPS"    | bc -l)" \
        "$(echo "scale=6; $sum_total/$REPS" | bc -l)" \
        "$(echo "scale=6; $sum_dp/$REPS"    | bc -l)"
}


echo "--- Πείραμα 1: Κλιμάκωση διεργασιών (N=2000, sparsity=90%, iters=10) ---"
CSV1="$OUTDIR/exp1_processes.csv"
echo "processes,csr_build,distribution,csr_spmv_serial,csr_spmv_par,total_csr,dense_spmv_par,speedup_spmv" > "$CSV1"

for P in 1 2 4 8; do
    read avg_b avg_d avg_cs avg_cp avg_tot avg_dp < <(run_avg 2000 90 10 "$P")
    sp_spmv=$(echo "scale=3; $avg_cs / $avg_cp" | bc -l)
    echo "$P,$avg_b,$avg_d,$avg_cs,$avg_cp,$avg_tot,$avg_dp,$sp_spmv" >> "$CSV1"
    printf "  processes=%-2d  build=%.4f  dist=%.4f  spmv_serial=%.4f  spmv_par=%.4f  total_csr=%.4f  dense_par=%.4f  speedup=%.2fx\n" \
           "$P" "$avg_b" "$avg_d" "$avg_cs" "$avg_cp" "$avg_tot" "$avg_dp" "$sp_spmv"
done
echo "  -> $CSV1"

echo ""
echo "--- Πείραμα 2: Επίδραση sparsity (N=2000, processes=4, iters=10) ---"
CSV2="$OUTDIR/exp2_sparsity.csv"
echo "sparsity,nnz,csr_build,csr_spmv_par,dense_spmv_par,total_csr" > "$CSV2"

for SP in 0 25 50 70 80 90 95 99; do
    nnz=$(mpirun --oversubscribe -np 1 ./mpi_spmv 2000 "$SP" 1 | awk -F'nnz=' '/^CSR_BUILD/{print $2}')
    read avg_b avg_d avg_cs avg_cp avg_tot avg_dp < <(run_avg 2000 "$SP" 10 4)
    echo "$SP,$nnz,$avg_b,$avg_cp,$avg_dp,$avg_tot" >> "$CSV2"
    printf "  sparsity=%-3d%%  nnz=%-8s  build=%.4f  csr=%.4f  dense=%.4f  csr_total=%.4f\n" \
           "$SP" "$nnz" "$avg_b" "$avg_cp" "$avg_dp" "$avg_tot"
done
echo "  -> $CSV2"


echo ""
echo "--- Πείραμα 3: Επίδραση μεγέθους πίνακα (sparsity=90%, processes=4, iters=5) ---"
CSV3="$OUTDIR/exp3_size.csv"
echo "N,csr_build,csr_spmv_par,dense_spmv_par,total_csr" > "$CSV3"

for N in 500 1000 2000 3000 5000; do
    read avg_b avg_d avg_cs avg_cp avg_tot avg_dp < <(run_avg "$N" 90 5 4)
    echo "$N,$avg_b,$avg_cp,$avg_dp,$avg_tot" >> "$CSV3"
    printf "  N=%-5d  build=%.4f  csr=%.4f  dense=%.4f  csr_total=%.4f\n" \
           "$N" "$avg_b" "$avg_cp" "$avg_dp" "$avg_tot"
done
echo "  -> $CSV3"

echo ""
echo "--- Πείραμα 4: Επίδραση επαναλήψεων (N=2000, sparsity=90%, processes=4) ---"
CSV4="$OUTDIR/exp4_iters.csv"
echo "iters,csr_build,csr_spmv_par,dense_spmv_par,total_csr" > "$CSV4"

for IT in 1 3 5 8 10 15 20; do
    read avg_b avg_d avg_cs avg_cp avg_tot avg_dp < <(run_avg 2000 90 "$IT" 4)
    echo "$IT,$avg_b,$avg_cp,$avg_dp,$avg_tot" >> "$CSV4"
    printf "  iters=%-3d  build=%.4f  csr=%.4f  dense=%.4f  csr_total=%.4f\n" \
           "$IT" "$avg_b" "$avg_cp" "$avg_dp" "$avg_tot"
done
echo "  -> $CSV4"

echo ""
echo "--- Παραγωγή γραφημάτων ---"

python3 - <<'PYEOF'
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = "results"

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
})

# ── Γράφημα 1: Κλιμάκωση διεργασιών ─────────────────────────────────────────
rows    = read_csv(f"{OUTDIR}/exp1_processes.csv")
procs   = [int(r['processes'])           for r in rows]
spmv_s  = [float(r['csr_spmv_serial'])   for r in rows]
spmv_p  = [float(r['csr_spmv_par'])      for r in rows]
dense_p = [float(r['dense_spmv_par'])    for r in rows]
dist_p  = [float(r['distribution'])      for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.plot(procs, spmv_s,  'o--', color='gray',    label='SpMV CSR serial (rank 0)')
ax1.plot(procs, spmv_p,  'o-',  color='#1f77b4', label='SpMV CSR parallel (MPI)')
ax1.plot(procs, dense_p, 's-',  color='#ff7f0e', label='SpMV Dense parallel (MPI)')
ax1.plot(procs, dist_p,  '^:',  color='#2ca02c', label='Distribution time')
ax1.set_xlabel('Αριθμός διεργασιών MPI')
ax1.set_ylabel('Χρόνος (s)')
ax1.set_title('Χρόνοι εκτέλεσης vs διεργασίες\n(N=2000, sparsity=90%, iters=10)')
ax1.set_xticks(procs)
ax1.legend(fontsize=8)

speedup_spmv  = [spmv_s[0] / p for p in spmv_p]
speedup_dense = [dense_p[0] / p for p in dense_p]
ax2.plot(procs, speedup_spmv,  'o-', color='#1f77b4', label='CSR SpMV speedup')
ax2.plot(procs, speedup_dense, 's-', color='#ff7f0e', label='Dense SpMV speedup')
ax2.plot(procs, procs, 'k--', linewidth=0.8, label='Ιδανικό speedup')
ax2.set_xlabel('Αριθμός διεργασιών MPI')
ax2.set_ylabel('Speedup')
ax2.set_title('Speedup vs διεργασίες\n(σχέση με σειριακή εκτέλεση)')
ax2.set_xticks(procs)
ax2.legend(fontsize=8)

fig.tight_layout()
fig.savefig(f"{OUTDIR}/plot1_processes.png")
plt.close()
print(f"  -> {OUTDIR}/plot1_processes.png")

# ── Γράφημα 2: Sparsity ─────────────────────────────────────────────────────
rows       = read_csv(f"{OUTDIR}/exp2_sparsity.csv")
sparsity   = [int(r['sparsity'])         for r in rows]
csr_build  = [float(r['csr_build'])      for r in rows]
csr_spmv   = [float(r['csr_spmv_par'])   for r in rows]
dense_spmv = [float(r['dense_spmv_par']) for r in rows]
csr_total  = [float(r['total_csr'])      for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.plot(sparsity, csr_spmv,   'o-', color='#1f77b4', label='CSR SpMV (parallel)')
ax1.plot(sparsity, dense_spmv, 's-', color='#ff7f0e', label='Dense SpMV (parallel)')
ax1.plot(sparsity, csr_build,  '^:', color='#2ca02c', label='CSR build (rank 0)')
ax1.set_xlabel('Sparsity (%)')
ax1.set_ylabel('Χρόνος (s)')
ax1.set_title('Χρόνοι SpMV vs sparsity\n(N=2000, processes=4, iters=10)')
ax1.legend(fontsize=8)

ax2.plot(sparsity, csr_total,  'o-', color='#1f77b4', label='CSR (build + distribution + SpMV)')
ax2.plot(sparsity, dense_spmv, 's-', color='#ff7f0e', label='Dense SpMV')
ax2.set_xlabel('Sparsity (%)')
ax2.set_ylabel('Χρόνος (s)')
ax2.set_title('CSR συνολικός χρόνος vs Dense')
ax2.legend(fontsize=8)

fig.tight_layout()
fig.savefig(f"{OUTDIR}/plot2_sparsity.png")
plt.close()
print(f"  -> {OUTDIR}/plot2_sparsity.png")

# ── Γράφημα 3: Μέγεθος πίνακα ───────────────────────────────────────────────
rows       = read_csv(f"{OUTDIR}/exp3_size.csv")
sizes      = [int(r['N'])                for r in rows]
csr_build  = [float(r['csr_build'])      for r in rows]
csr_spmv   = [float(r['csr_spmv_par'])   for r in rows]
dense_spmv = [float(r['dense_spmv_par']) for r in rows]
csr_total  = [float(r['total_csr'])      for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.plot(sizes, csr_spmv,   'o-', color='#1f77b4', label='CSR SpMV')
ax1.plot(sizes, dense_spmv, 's-', color='#ff7f0e', label='Dense SpMV')
ax1.plot(sizes, csr_build,  '^:', color='#2ca02c', label='CSR build')
ax1.set_xlabel('N (διάσταση πίνακα)')
ax1.set_ylabel('Χρόνος (s)')
ax1.set_title('Χρόνοι vs μέγεθος πίνακα\n(sparsity=90%, processes=4, iters=5)')
ax1.legend(fontsize=8)

ax2.plot(sizes, csr_total,  'o-', color='#1f77b4', label='CSR (build + distribution + SpMV)')
ax2.plot(sizes, dense_spmv, 's-', color='#ff7f0e', label='Dense SpMV')
ax2.set_xlabel('N (διάσταση πίνακα)')
ax2.set_ylabel('Χρόνος (s)')
ax2.set_title('CSR συνολικός vs Dense')
ax2.legend(fontsize=8)

fig.tight_layout()
fig.savefig(f"{OUTDIR}/plot3_size.png")
plt.close()
print(f"  -> {OUTDIR}/plot3_size.png")

# ── Γράφημα 4: Αριθμός επαναλήψεων ──────────────────────────────────────────
rows       = read_csv(f"{OUTDIR}/exp4_iters.csv")
iters      = [int(r['iters'])            for r in rows]
csr_build  = [float(r['csr_build'])      for r in rows]
csr_spmv   = [float(r['csr_spmv_par'])   for r in rows]
dense_spmv = [float(r['dense_spmv_par']) for r in rows]
csr_total  = [float(r['total_csr'])      for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.plot(iters, csr_spmv,   'o-', color='#1f77b4', label='CSR SpMV')
ax1.plot(iters, dense_spmv, 's-', color='#ff7f0e', label='Dense SpMV')
ax1.axhline(y=csr_build[0], color='#2ca02c', linestyle='--',
            linewidth=0.9, label='CSR build (σταθερό)')
ax1.set_xlabel('Αριθμός επαναλήψεων')
ax1.set_ylabel('Χρόνος (s)')
ax1.set_title('Χρόνοι SpMV vs επαναλήψεις\n(N=2000, sparsity=90%, processes=4)')
ax1.legend(fontsize=8)

ax2.plot(iters, csr_total,  'o-', color='#1f77b4', label='CSR (build + distribution + SpMV)')
ax2.plot(iters, dense_spmv, 's-', color='#ff7f0e', label='Dense SpMV')
for i in range(len(iters) - 1):
    if csr_total[i] >= dense_spmv[i] and csr_total[i + 1] < dense_spmv[i + 1]:
        mid = (iters[i] + iters[i + 1]) / 2
        ax2.axvline(x=mid, color='red', linestyle=':', linewidth=1.2, label='Crossover')
        break
ax2.set_xlabel('Αριθμός επαναλήψεων')
ax2.set_ylabel('Χρόνος (s)')
ax2.set_title('CSR συνολικός vs Dense — crossover\n(build+distribution αποσβένεται με επαναλήψεις)')
ax2.legend(fontsize=8)

fig.tight_layout()
fig.savefig(f"{OUTDIR}/plot4_iters.png")
plt.close()
print(f"  -> {OUTDIR}/plot4_iters.png")

print("\nΌλα τα γραφήματα αποθηκεύτηκαν στο φάκελο results/")
PYEOF

echo ""
echo "=== Ολοκληρώθηκε ==="
echo "Αρχεία στο $OUTDIR/:"
ls -lh "$OUTDIR/"