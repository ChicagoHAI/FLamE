PET_DIR=/data/rosa/flame #TODO: point this to your flame directory

DATA_DIR=../data/e-SNLI-k16 #TODO: point this to your data directory
MODEL_TYPE=roberta
MODEL=roberta-large
TASK=snli

# -1 => Use everything
NUM_TRAIN=-1
NUM_TEST=-1
NUM_UNLABEL=-1

export CUDA_VISIBLE_DEVICES=1

MSG=esnli_k16_pet_oracle
python3 $PET_DIR/cli.py \
--method pet \
--data_dir $DATA_DIR \
--model_type $MODEL_TYPE \
--model_name_or_path $MODEL \
--task_name $TASK \
--output_dir $PET_DIR/outputs/$TASK/pet_$TASK_$MSG \
--do_train \
--do_eval \
--overwrite_output_dir \
--cache_dir $PET_DIR/cache/$TASK/$MODEL/ \
--train_examples $NUM_TRAIN \
--test_examples $NUM_TEST \
--unlabeled_examples $NUM_UNLABEL \
--no_distillation \
--eval_result $MSG \
--wandb_run_name pet_$TASK_$MSG \
--e_pet_test \
--e_pet_pred \
--train_gold_expl \
--test_gold_expl \
--pattern_ids 0 1 2 3 \
--pet_max_steps 1000 \
--pet_per_gpu_train_batch_size 1 \
--pet_per_gpu_unlabeled_batch_size 3 \
--pet_gradient_accumulation_steps 4