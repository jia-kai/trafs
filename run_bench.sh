#!/bin/bash -e

set -u -e

declare -a PROBS=( SPL DPL LLR LLC DG )
BENCH_SIZE=50
OUTDIR=bench-out

# reserve some contigous CPUs for each task to minimize cache contention
CPU_PER_TASK=6
# only use first two CPUs in affinity setting
CPU_USE_PER_TASK=4

nr_cpu=$(nproc)
nr_par=$((nr_cpu / CPU_PER_TASK))
echo "Running $nr_par parallel tasks on $nr_cpu CPUs"

mkdir -p $OUTDIR

parallel -j $nr_par --colsep '-' --lb --eta --progress "$@" \
    --argfile=<(
for i in  "${PROBS[@]}"; do
    for j in $(seq 0 $(( $BENCH_SIZE-1 ))); do
        printf '%s-%02d\n' $i $j
    done
done) bash -c "\" \
[ -f $OUTDIR/{1}.{2}.pkl ] && exit 0; \
stdbuf -i0 -o0 -e0 \
    taskset --cpu-list \
    \$(( ({%}-1)*$CPU_PER_TASK ))-\$(( ({%}-1)*$CPU_PER_TASK+$CPU_USE_PER_TASK-1 )) \
    ./run_bench.py -o $OUTDIR/{1}.{2}.pkl -s $BENCH_SIZE \
        -p {1} -i {2} 2>&1 | tee $OUTDIR/{1}.{2}.log \
\""
