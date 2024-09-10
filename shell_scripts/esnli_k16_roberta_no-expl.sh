PET_DIR=/data/rosa/flame #TODO: point this to your flame directory

DATA_DIR=../data/e-SNLI-k16 #TODO: point this to your data directory
MODEL_TYPE=roberta
MODEL=roberta-large
TASK=esnli

# -1 => Use everything
NUM_TRAIN=-1
NUM_TEST=-1
NUM_UNLABEL=-1

export CUDA_VISIBLE_DEVICES=0

MSG=esnli_k16_roberta_no-expl
python3 $PET_DIR/cli.py \
--method sequence_classifier \
--data_dir $DATA_DIR \
--model_type $MODEL_TYPE \
--model_name_or_path $MODEL \
--task_name $TASK \
--output_dir $PET_DIR/outputs/$TASK/sequence_classifier/$MSG \
--do_train \
--do_eval \
--overwrite_output_dir \
--cache_dir $PET_DIR/cache/$TASK/$MODEL/ \
--train_examples $NUM_TRAIN \
--test_examples $NUM_TEST \
--unlabeled_examples $NUM_UNLABEL \
--no_distillation \
--eval_result $MSG \
--wandb_run_name sequence_classifier_$TASK_$MSG \
--save_train_logits \
--eval_set test \
--sc_max_steps 1000 