#!/bin/bash --login
#PBS -l select=256:system=polaris
#PBS -l walltime=24:00:00
#PBS -l filesystems=home:eagle:grand
#PBS -q prod
#PBS -A SkillfulWeather
#PBS -j oe
#PBS -o /lus/eagle/projects/MDClimSim/jstock/logs
#PBS -N era5-diffusion-layers

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
hyper_name="layers"
LOCAL_HP_DEPTH=(12 8)
LOCAL_HP_DIM=(1056 768)
LOCAL_HP_HEADS=(12 8)

# --------------------------------
# main loop
node_counter=0

for local_hp_depth in "${LOCAL_HP_DEPTH[@]}"; do
  for local_hp_dim in "${LOCAL_HP_DIM[@]}"; do
    for local_hp_heads in "${LOCAL_HP_HEADS[@]}"; do

      content=$(head -n $((node_counter + NODES_PER_JOB)) "${PBS_NODEFILE}" | tail -n ${NODES_PER_JOB})
      file=$(mktemp /tmp/ezpz.XXXXXXXX)
      echo "${content}" > "${file}"
      cat "${file}"
      ezpz_setup_job "${file}"

      experiment="era5-swinv2-1.4-trigflow"
      experiment_name="era5-swinv2-1.4-trigflow-${hyper_name}-${local_hp_depth}-${local_hp_dim}-${local_hp_heads}"
      echo "Starting ${experiment_name}..."

      run_cmd="${DIST_LAUNCH} python3 -m swift.train \
          experiment=${experiment} \
          experiment_name=${experiment_name} \
          data.batch_size=${BATCH_SIZE} \
          model.depth=${local_hp_depth} \
          model.dim=${local_hp_dim} \
          model.heads=${local_hp_heads} \
          resume=${resume}"

      echo "${run_cmd}\n"
      eval "${run_cmd}" &
      node_counter=$((node_counter + NODES_PER_JOB))

    done
  done
done

echo "needs ${node_counter} hosts."
wait