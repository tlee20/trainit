# Script for batch-submitting LR fine-tuning

#-------------------------------------------------------------------------
# Some global variables and the `submit_job()` function

BASE_DIR=/projectnb/aclab/alee12/trainit3/trainit_project/trainit  # change to your path here; you can just copy from your config.sh
EXP=iter10k_firstseg_finetune   # experiment name; you can change if you want
PROJECT=test3                     # change to your wandb project name
TOTAL_STEPS=1000                # maximum number of training steps
# NOTE: for now, I just set this value of be 10% of the total steps (which is 0.1 * 10k = 1000).
# Pls make sure in your later experiment, keep the size of the first segment the same as this value
lrs=(1e-1 1e-2 1e-3 1e-4 1e-5)  # log grid of LRs
# lrs=(0.33e-X 0.50e-X 0.67e-X 1.50e-X 2.00e-X 3.00e-X) # linear grid
#   NOTE: once you find the optimal LR in the log grid, comment out the log grid
#   and uncomment the linear grid, with X replaced with the optimal LR.
#   e.g., if 1e-3 is optimal in log grid, change to 0.33e-3, 0.50e-3, ...


NODES=1
GPU="L40S"
TIME="12:00:00"
DATE=$(date +"%Y-%m-%d")
OUTPUT_PATH=$BASE_DIR/scheduler_outputs/$DATE/$EXP
SCC_PATH=$OUTPUT_PATH/scc_outputs

mkdir -p $OUTPUT_PATH
mkdir -p $SCC_PATH

submit_job() {
    local args=("$@")
    job_output=$(qsub <<EOF
#!/bin/bash -l
#$ -pe omp 8
#$ -l h="!scc-506"          # Blacklists bad nodes
#$ -l gpus=${NODES}
#$ -l gpu_type=${GPU}       # Specifies the gpu type
#$ -l h_rt=${TIME}          # Specifies the hard time limit for the job
#$ -N "$name"
#$ -o $OUTPUT_PATH/\$JOB_NAME.o\$JOB_ID
#$ -e $OUTPUT_PATH/\$JOB_NAME.e\$JOB_ID
cd ${BASE_DIR}
source activate_env.sh
python main.py ${args[@]}
EOF
    )
    # Extract job id and log the submission.
    job_id=$(echo "$job_output" | awk '{print $3}')
    echo "$(date '+%Y-%m-%d %H:%M:%S') job_id: ${job_id} || ${name}" >> "${OUTPUT_PATH}/job_list.txt"
    echo "Submitted job: $name"
}

#-------------------------------------------------------------------------
# Other training configs. No need to change.

# mixed precision
USE_AMP=False

# training batch size
BATCH_SIZE=128

# optimizer configs
OPTIMIZER=adamw
BETA1=0.9
BETA2=0.999
WEIGHT_DECAY=0.1
NESTEROV=False

# log additional metrics to wandb
LOG_CALLBACK_DATA=False

for lr in "${lrs[@]}"; do
    name="${EXP}_lr2${lr}"
    args=(
        "logging.wandb_project=$PROJECT"
        "logging.wandb_name=$name"
        "logging.wandb_runid=$(uuidgen)"
        "logging.wandb_expname=$name"
        "logging.log_callback_data=$LOG_CALLBACK_DATA"
        "train.max_steps=$TOTAL_STEPS"
        "train.use_amp=$USE_AMP"
        "dataset.total_batch_size=$BATCH_SIZE"
        "random_seed=$SEED"
        "optimizer=$OPTIMIZER"
        "optimizer.beta1=$BETA1"
        "optimizer.beta2=$BETA2"
        "optimizer.weight_decay=$WEIGHT_DECAY"
        "optimizer.use_nesterov=$NESTEROV"
    )
    # schedule configs
    args+=(
        "optimizer/lr_config=piecewise_linear"
        "optimizer.lr_config.lr1=0"
        "optimizer.lr_config.lr2=$lr"
        "optimizer.lr_config.start_steps=0"
        "optimizer.lr_config.max_steps=$TOTAL_STEPS"
    )
    submit_job ${args[@]}
done