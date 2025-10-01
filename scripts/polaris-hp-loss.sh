#!/bin/bash --login
#PBS -l select=256:system=polaris
#PBS -l walltime=24:00:00
#PBS -l filesystems=home:eagle:grand
#PBS -q prod
#PBS -A SkillfulWeather
#PBS -j oe
#PBS -o /lus/eagle/projects/MDClimSim/jstock/logs
#PBS -N era5-diffusion

echo "Job started at: $(date '+%Y-%m-%d-%H%M%S')"

cd /eagle/projects/MDClimSim/jstock/prod/swift

source /etc/profile
module use /soft/modulefiles; module load conda; conda activate base
source /eagle/projects/MDClimSim/jstock/prod/swift/venv/bin/activate

source <(curl -s https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh)

# --------------------------------
# resume setup (using PARTID from environment)
: ${PARTID:=0} # default to 0
if ! [[ "$PARTID" =~ ^[0-9]+$ ]]; then
  echo "Error: PARTID must be a non-negative integer." >&2
  exit 1
fi
PARTID=$(printf "%03d" "$PARTID")
if [[ "$PARTID" == "000" ]]; then
  resume=null
else
  resume=$(printf "%03d" $((10#$PARTID - 1)))
fi
echo "PARTID: $PARTID"

export HYDRA_FULL_ERROR=1
export HYDRA_RUN_ID=$PARTID

# --------------------------------
# job parallelism
NODES_PER_JOB=32
NGPU_PER_HOST=$(nvidia-smi -L | wc -l)
BATCH_SIZE=$((NGPU_PER_HOST * NODES_PER_JOB * 1))

# --------------------------------
# hyperparameters (total 8)
hyper_name="lognormal"
MEANS=(-0.8 -0.2)
STDS=(1.6 2.2 2.8 3.2)

# --------------------------------
# main loop
node_counter=0

for mean in "${MEANS[@]}"; do
  for std in "${STDS[@]}"; do

    content=$(head -n $((node_counter + NODES_PER_JOB)) "${PBS_NODEFILE}" | tail -n ${NODES_PER_JOB})
    file=$(mktemp /tmp/ezpz.XXXXXXXX)
    echo "${content}" > "${file}"
    cat "${file}"
    ezpz_setup_job "${file}"

    experiment="era5-swinv2-1.4-trigflow"
    experiment_name="era5-swinv2-1.4-trigflow-${hyper_name}-${mean}-${std}"
    echo "Starting ${experiment_name}..."

    run_cmd="${DIST_LAUNCH} python3 -m swift.train \
        experiment=${experiment} \
        experiment_name=${experiment_name} \
        data.batch_size=${BATCH_SIZE} \
        loss.P_mean=${mean} \
        loss.P_std=${std} \
        resume=${resume}"

    echo "${run_cmd}\n"
    eval "${run_cmd}" &
    node_counter=$((node_counter + NODES_PER_JOB))

  done
done

echo "needs ${node_counter} hosts."
wait