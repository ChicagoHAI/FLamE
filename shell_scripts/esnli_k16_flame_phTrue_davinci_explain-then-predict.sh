PET_DIR=/data/rosa/flame #TODO: point this to your flame directory

DATA_DIR=../data/e-SNLI-k16 #TODO: point this to your data directory
MODEL_TYPE=roberta
MODEL=roberta-large
TASK=esnli

# -1 => Use everything
NUM_TRAIN=-1
NUM_TEST=-1
NUM_UNLABEL=-1

export CUDA_VISIBLE_DEVICES=1

MSG=esnli_k16_flame_phTrue_davinci_explain-then-predict
python3 $PET_DIR/cli.py \
--method pet \
--pattern_ids 0 1 2 3 \
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
--pet_max_steps 1000 \
--pet_per_gpu_train_batch_size 1 \
--pet_per_gpu_unlabeled_batch_size 3 \
--pet_gradient_accumulation_steps 4 \
--no_distillation \
--eval_result $MSG \
--wandb_run_name pet_$TASK_$MSG \
--train_custom_expl_file $DATA_DIR/phel_expl/train_davinci_phel.jsonl \
--dev_custom_expl_file $DATA_DIR/phel_expl/dev_davinci_phel.jsonl \
--test_custom_expl_file $DATA_DIR/phel_expl/test_davinci_phel.jsonl \
--eval_with_three_labels_explanations_logits \
--e_pet_pred \
--e_pet_test \
--train_gold_expl \
--train_with_three_labels_explanations \
--calibration \
--beta_requires_grad \
--beta 0.5 \
--beta_lr 2e-3