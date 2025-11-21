wandb offline
# Set your API keys as environment variables before running this script
# export OPENAI_API_KEY="your_openai_api_key_here"
# export WANDB_API_KEY="your_wandb_api_key_here"
export CUDA_VISIBLE_DEVICES="0";
export HF_HOME="/workspace/cache"
# export HF_ACCESS_TOKEN="your_huggingface_token_here"
export WANDB=T


### Experiment type ###
export RUN_GEN=T; export RUN_ATT=F; export RUN_EVAL=T; export DEBUG=F;
export LIMIT_ROWS=-1


### Generation ###
export MODEL_PATH="meta-llama/Llama-2-7b-hf"; export BS=1; export TOKEN_LEN=50;
export D_NAME="c4"; export D_CONFIG="realnewslike"; export INPUT_FILTER="prompt_and_completion_length"
if [ $D_NAME == "lfqa" ]
then
  export INPUT_FILTER="completion_length"
fi
export NUM_BEAMS=1; export SAMPLING=T
export FP16=T; export MIN_GEN=3
## multi-bit
export MSG_LEN=4; export CODE_LEN=4; export RADIX=4; export ZERO_BIT=F;
export FEEDBACK=F; export F_BIAS=-1; export F_TAU=1; export F_ETA=3
export USE_PPRF=F; export USE_FIXP=F;
export SEED_SCH="lefthash"; export GAMMA=0.25; export DELTA=2.0;
export ADP_SAMPLE=F
## logging
export RUN_NAME="fixedT-${MSG_LEN}b-250T-${RADIX}R-0.25GAMMA-lefthash"
export RUN_NAME="fixedT-1b-250T-2R-0.25GAMMA-lefthash"
#export OUTPUT_DIR="./experiments/llama-7b/adaptive-bias"
export OUTPUT_DIR="./experiments/llama-7b/"


### Attack ###
export ATTACK_M="dipper"; export DIPPER_ORDER=0; export DIPPER_LEX=60;
export ATTACK_M="copy-paste"; export srcp="50%"; export CP_ATT_TYPE="single-single"
export ATTACK_SUFFIX="cp=.5"


### Evaluation ###
export LOWER_TOL=25; export UPPER_TOL=25
export ORACLE_MODEL="meta-llama/Llama-2-13b-hf"
export IGNORE_R_NGRAM=T
export EVAL_METRICS="z-score"


mkdir -p ${OUTPUT_DIR}/log/${RUN_NAME}
bash ./run_pipeline.sh 2>&1 | tee -a ${OUTPUT_DIR}/log/${RUN_NAME}/output.log
cat ${OUTPUT_DIR}/log/${RUN_NAME}/output.log | grep "auc"
