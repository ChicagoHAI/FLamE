# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script can be used to train and evaluate either a regular supervised model or a PET/iPET model on
one of the supported tasks and datasets.
"""

import argparse
import os
from re import template
from typing import Tuple

import torch
import wandb
from pet.tasks import ADDITIONAL_TEST_SET, PROCESSORS, load_examples, UNLABELED_SET, TRAIN_SET, DEV_SET, TEST_SET, METRICS, DEFAULT_METRICS
from pet.utils import eq_div
from pet.wrapper import WRAPPER_TYPES, MODEL_CLASSES, SEQUENCE_CLASSIFIER_WRAPPER, WrapperConfig
from pet.slurm_utils import setup_slurm
import pet
import log


logger = log.get_logger('root')


def load_pet_configs(args) -> Tuple[WrapperConfig, pet.TrainConfig, pet.EvalConfig]:
    """
    Load the model, training and evaluation configs for PET from the given command line arguments.
    """
    model_cfg = WrapperConfig(model_type=args.model_type, model_name_or_path=args.model_name_or_path,
                              wrapper_type=args.wrapper_type, task_name=args.task_name, label_list=args.label_list,
                              max_seq_length=args.pet_max_seq_length, verbalizer_file=args.verbalizer_file,
                              cache_dir=args.cache_dir, beta=args.beta, beta_requires_grad=args.beta_requires_grad,
                              beta_lr=args.beta_lr,
                              calibration_mode=args.calibration_mode, calibrate_on_step=args.calibrate_on_step, calibrate_on_end=args.calibrate_on_end)

    train_cfg = pet.TrainConfig(device=args.device, per_gpu_train_batch_size=args.pet_per_gpu_train_batch_size,
                                per_gpu_unlabeled_batch_size=args.pet_per_gpu_unlabeled_batch_size, n_gpu=args.n_gpu,
                                num_train_epochs=args.pet_num_train_epochs, max_steps=args.pet_max_steps,
                                gradient_accumulation_steps=args.pet_gradient_accumulation_steps,
                                weight_decay=args.weight_decay, learning_rate=args.learning_rate,
                                adam_epsilon=args.adam_epsilon, warmup_steps=args.warmup_steps,
                                max_grad_norm=args.max_grad_norm, lm_training=args.lm_training, alpha=args.alpha,
                                train_with_all_expl=args.train_with_three_labels_explanations)

    eval_cfg = pet.EvalConfig(device=args.device, n_gpu=args.n_gpu, metrics=args.metrics,
                              per_gpu_eval_batch_size=args.pet_per_gpu_eval_batch_size,
                              decoding_strategy=args.decoding_strategy, priming=args.priming,
                              eval_result=args.eval_result, 
                              eval_with_three_labels_explanations_logits=args.eval_with_three_labels_explanations_logits,
                              )

    return model_cfg, train_cfg, eval_cfg


def load_sequence_classifier_configs(args) -> Tuple[WrapperConfig, pet.TrainConfig, pet.EvalConfig]:
    """
    Load the model, training and evaluation configs for a regular sequence classifier from the given command line
    arguments. This classifier can either be used as a standalone model or as the final classifier for PET/iPET.
    """
    model_cfg = WrapperConfig(model_type=args.model_type, model_name_or_path=args.model_name_or_path,
                              wrapper_type=SEQUENCE_CLASSIFIER_WRAPPER, task_name=args.task_name,
                              label_list=args.label_list, max_seq_length=args.sc_max_seq_length,
                              verbalizer_file=args.verbalizer_file, cache_dir=args.cache_dir, beta_requires_grad=args.beta_requires_grad)

    train_cfg = pet.TrainConfig(device=args.device, per_gpu_train_batch_size=args.sc_per_gpu_train_batch_size,
                                per_gpu_unlabeled_batch_size=args.sc_per_gpu_unlabeled_batch_size, n_gpu=args.n_gpu,
                                num_train_epochs=args.sc_num_train_epochs, max_steps=args.sc_max_steps,
                                temperature=args.temperature,
                                gradient_accumulation_steps=args.sc_gradient_accumulation_steps,
                                weight_decay=args.weight_decay, learning_rate=args.learning_rate,
                                adam_epsilon=args.adam_epsilon, warmup_steps=args.warmup_steps,
                                max_grad_norm=args.max_grad_norm, use_logits=args.method != 'sequence_classifier',
                                sc_eval_during_train=args.sc_eval_during_train, sc_eval_steps=args.sc_eval_steps,
                                train_with_all_expl=args.train_with_three_labels_explanations)

    eval_cfg = pet.EvalConfig(device=args.device, n_gpu=args.n_gpu, metrics=args.metrics,
                              per_gpu_eval_batch_size=args.sc_per_gpu_eval_batch_size,
                              eval_result=args.eval_result, 
                              eval_with_three_labels_explanations_logits=args.eval_with_three_labels_explanations_logits,
                              )

    return model_cfg, train_cfg, eval_cfg


def load_ipet_config(args) -> pet.IPetConfig:
    """
    Load the iPET config from the given command line arguments.
    """
    ipet_cfg = pet.IPetConfig(generations=args.ipet_generations, logits_percentage=args.ipet_logits_percentage,
                              scale_factor=args.ipet_scale_factor, n_most_likely=args.ipet_n_most_likely)
    return ipet_cfg


def main():
    setup_slurm()
    parser = argparse.ArgumentParser(description="Command line interface for PET/iPET")
    # Required parameters
    parser.add_argument("--method", required=True, choices=['pet', 'ipet', 'sequence_classifier'],
                        help="The training method to use. Either regular sequence classification, PET or iPET.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True, choices=MODEL_CLASSES.keys(),
                        help="The type of the pretrained language model to use")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--task_name", default=None, type=str, required=True, choices=PROCESSORS.keys(),
                        help="The name of the task to train/evaluate on")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written")

    # PET-specific optional parameters
    parser.add_argument("--wrapper_type", default="mlm", choices=WRAPPER_TYPES,
                        help="The wrapper type. Set this to 'mlm' for a masked language model like BERT or to 'plm' "
                             "for a permuted language model like XLNet (only for PET)")
    parser.add_argument("--pattern_ids", default=[0], type=int, nargs='+',
                        help="The ids of the PVPs to be used (only for PET)")
    parser.add_argument("--lm_training", action='store_true',
                        help="Whether to use language modeling as auxiliary task (only for PET)")
    parser.add_argument("--alpha", default=0.9999, type=float,
                        help="Weighting term for the auxiliary language modeling task (only for PET)")
    parser.add_argument("--temperature", default=2, type=float,
                        help="Temperature used for combining PVPs (only for PET)")
    parser.add_argument("--verbalizer_file", default=None,
                        help="The path to a file to override default verbalizers (only for PET)")
    parser.add_argument("--reduction", default='wmean', choices=['wmean', 'mean'],
                        help="Reduction strategy for merging predictions from multiple PET models. Select either "
                             "uniform weighting (mean) or weighting based on train set accuracy (wmean)")
    parser.add_argument("--decoding_strategy", default='default', choices=['default', 'ltr', 'parallel'],
                        help="The decoding strategy for PET with multiple masks (only for PET)")
    parser.add_argument("--no_distillation", action='store_true',
                        help="If set to true, no distillation is performed (only for PET)")
    parser.add_argument("--pet_repetitions", default=3, type=int,
                        help="The number of times to repeat PET training and testing with different seeds.")
    parser.add_argument("--pet_max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization for PET. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--pet_per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for PET training.")
    parser.add_argument("--pet_per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for PET evaluation.")
    parser.add_argument("--pet_per_gpu_unlabeled_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for auxiliary language modeling examples in PET.")
    parser.add_argument('--pet_gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass in PET.")
    parser.add_argument("--pet_num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform in PET.")
    parser.add_argument("--pet_max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform in PET. Override num_train_epochs.")

    # SequenceClassifier-specific optional parameters (also used for the final PET classifier)
    parser.add_argument("--sc_repetitions", default=1, type=int,
                        help="The number of times to repeat seq. classifier training and testing with different seeds.")
    parser.add_argument("--sc_max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization for sequence classification. "
                             "Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--sc_per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for sequence classifier training.")
    parser.add_argument("--sc_per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for sequence classifier evaluation.")
    parser.add_argument("--sc_per_gpu_unlabeled_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for unlabeled examples used for distillation.")
    parser.add_argument('--sc_gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass for "
                             "sequence classifier training.")
    parser.add_argument("--sc_num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform for sequence classifier training.")
    parser.add_argument("--sc_max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform for sequence classifier training. "
                             "Override num_train_epochs.")
    parser.add_argument("--sc_eval_during_train", action='store_true',
                        help="Whether to evaluate during training for sequence classifier.")
    parser.add_argument("--sc_eval_steps", default=-1, type=int,
                        help="Evaluate after every this number of steps during training for sequence classifier.")
    parser.add_argument("--sc_phe", action='store_true',
                        help="whether to use explanation in the sequence classifier input sequence. For example [cls]p[sep]h[sep]e[sep]")

    # iPET-specific optional parameters
    parser.add_argument("--ipet_generations", default=3, type=int,
                        help="The number of generations to train (only for iPET)")
    parser.add_argument("--ipet_logits_percentage", default=0.25, type=float,
                        help="The percentage of models to choose for annotating new training sets (only for iPET)")
    parser.add_argument("--ipet_scale_factor", default=5, type=float,
                        help="The factor by which to increase the training set size per generation (only for iPET)")
    parser.add_argument("--ipet_n_most_likely", default=-1, type=int,
                        help="If >0, in the first generation the n_most_likely examples per label are chosen even "
                             "if their predicted label is different (only for iPET)")

    # Other optional parameters
    parser.add_argument("--train_examples", default=-1, type=int,
                        help="The total number of train examples to use, where -1 equals all examples.")
    parser.add_argument("--test_examples", default=-1, type=int,
                        help="The total number of test examples to use, where -1 equals all examples.")
    parser.add_argument("--unlabeled_examples", default=-1, type=int,
                        help="The total number of unlabeled examples to use, where -1 equals all examples")
    parser.add_argument("--split_examples_evenly", action='store_true',
                        help="If true, train examples are not chosen randomly, but split evenly across all labels.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where to store the pre-trained models downloaded from S3.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--do_train', action='store_true',
                        help="Whether to perform training")
    parser.add_argument('--do_eval', action='store_true',
                        help="Whether to perform evaluation")
    parser.add_argument('--priming', action='store_true',
                        help="Whether to use priming for evaluation")
    parser.add_argument("--eval_set", choices=['dev', 'test'], default='dev',
                        help="Whether to perform evaluation on the dev set or the test set")
    parser.add_argument("--additional_test", action='store_true',
                        help="Evaluate on an additional test set.")
    parser.add_argument("--no_dev_set", action='store_true',
                        help='Whether there is a dev set.')

    # explanations args
    parser.add_argument('--e_pet_pred', action='store_true',
                        help="Whether to use explanations patterns for verbalizer word prediction")
    parser.add_argument('--e_pet_lm', action='store_true',
                        help="Whether to use explanations patterns for mlm auxiliary task")
    parser.add_argument('--e_pet_test', action='store_true',
                        help="Whether to use explanations patterns for testing pet")
    parser.add_argument('--wandb_run_name', type=str, default=None, help="Name to display for this run on wandb")
    parser.add_argument('--save_train_logits', action='store_true', help="Whether to save the training logits")
    parser.add_argument('--train_custom_expl_file', type=str, default=None, help="Aligned custom explanations with training set")
    parser.add_argument('--dev_custom_expl_file', type=str, default=None, help="Aligned custom explanations with dev set")
    parser.add_argument('--test_custom_expl_file', type=str, default=None, help="Aligned custom explanations with test set")
    parser.add_argument('--additional_test_custom_expl_file', type=str, default=None, help="Aligned custom explanations with the additional test set")
    parser.add_argument('--concat_expl', action='store_true', help="Whether to concatenate explanations of all labels")
    parser.add_argument('--shuffle_expl_unlabeled', action='store_true', help="Whether to shuffle explanations in unlabeled set, requires concat_expl to be true")
    parser.add_argument('--shuffle_expl_all', action='store_true', help="Whether to shuffle explanations in train+unlabeled set, requires concat_expl to be true")
    parser.add_argument('--train_gold_expl', action='store_true', help="Whether to only use gold explanations in training")
    parser.add_argument('--train_gold_gen_expl', action='store_true', help="Whether to use only explanations generated with the gold label")
    parser.add_argument('--test_gold_expl', action='store_true', help='Whether to use gold explanations in testing')
    parser.add_argument('--templated_explanation', action='store_true', help='Whether gold training explanations are templated')

    # ensembling and calibration args
    parser.add_argument('--train_with_three_labels_explanations', action='store_true', 
                        help="Set to true to train each example with all three generated explanations and set the true label to be the one used for  \
                              explanation generation. (AKA an option in idea 1).")
    parser.add_argument('--eval_with_three_labels_explanations_logits', action='store_true', 
                        help="Set to true to evaluate each example with all three generated explanations and pick label with largest logit. (AKA idea 1).")
    parser.add_argument('--calibration', action='store_true', help='whether to use calibration (ph, phe logits) or not')
    parser.add_argument('--beta', type=float, default=1.0, help="beta value for combining logits with calibration logits." +
        " set to 1.0 to ignore calibration logits, and set to 0.0 to exclusively use calibration logits.")
    parser.add_argument('--beta_requires_grad', action='store_true', help="set to true to make beta a learnable parameter")
    parser.add_argument('--beta_lr', default=2e-3, type=float, help="learning rate for beta.")
    parser.add_argument('--calibration_mode', type=str, choices=['multiplicative', 'additive'],
            default="multiplicative", help='mode of calibration')
    parser.add_argument('--calibrate_on_step', action='store_true', help='calibrate model at every optimizer step')
    parser.add_argument('--calibrate_on_end', action='store_true', help='calibrate once when model finishes training')

    # saving customization args
    parser.add_argument('--eval_result', default=None, type=str,
                        help="Append to of eval results files. Good for when you evaluate on new data and wanna save under a different name.")

    args = parser.parse_args()
    logger.info("Parameters: {}".format(args))
    
    wandb.init(project="pet", config=args, id=wandb.util.generate_id(), name=args.wandb_run_name)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
            and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    if (args.shuffle_expl_unlabeled or args.shuffle_expl_all) and not args.concat_expl:
        raise Exception("Can only shuffle expl if concat_expl is true")

    if args.shuffle_expl_all:
        args.shuffle_expl_unlabeled = True

    # Setup CUDA, GPU & distributed training
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    args.n_gpu = torch.cuda.device_count()
    logger.info("n_gpu: {}".format(args.n_gpu))

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in PROCESSORS:
        raise ValueError("Task '{}' not found".format(args.task_name))
    processor = PROCESSORS[args.task_name]()
    args.label_list = processor.get_labels()

    train_ex_per_label, test_ex_per_label = None, None
    train_ex, test_ex = args.train_examples, args.test_examples
    if args.split_examples_evenly:
        train_ex_per_label = eq_div(args.train_examples, len(args.label_list)) if args.train_examples != -1 else -1
        test_ex_per_label = eq_div(args.test_examples, len(args.label_list)) if args.test_examples != -1 else -1
        train_ex, test_ex = None, None

    train_data = load_examples(
        args.task_name, args.data_dir, TRAIN_SET, num_examples=train_ex,
         num_examples_per_label=train_ex_per_label, no_expl=not args.e_pet_pred,
         expl_file=args.train_custom_expl_file, concat_expl=args.concat_expl, shuffle_expl=args.shuffle_expl_all,
         train_with_three_labels_explanations=args.train_with_three_labels_explanations,
         calibration=args.calibration, 
         train_gold_expl=args.train_gold_expl,
         train_gold_gen_expl=args.train_gold_gen_expl,
         templated_explanation=args.templated_explanation)

    if not args.no_dev_set:
        dev_data = load_examples(
            # i don't think it make sense to shuffle expl during eval
            args.task_name, args.data_dir, DEV_SET,
            num_examples=test_ex, num_examples_per_label=test_ex_per_label, no_expl=not args.e_pet_test,
            expl_file=args.dev_custom_expl_file, concat_expl=args.concat_expl, shuffle_expl=False,
            eval_with_three_labels_explanations_logits=args.eval_with_three_labels_explanations_logits,
            calibration=args.calibration,
            test_gold_expl=args.test_gold_expl)

    test_data = load_examples(
        # i don't think it make sense to shuffle expl during eval
        args.task_name, args.data_dir, TEST_SET,
         num_examples=test_ex, num_examples_per_label=test_ex_per_label, no_expl=not args.e_pet_test,
         expl_file=args.test_custom_expl_file, concat_expl=args.concat_expl, shuffle_expl=False,
         eval_with_three_labels_explanations_logits=args.eval_with_three_labels_explanations_logits,
         calibration=args.calibration,
         test_gold_expl=args.test_gold_expl)

    if args.additional_test:
        additional_test_data = load_examples(
            # i don't think it make sense to shuffle expl during eval
            args.task_name, args.data_dir, ADDITIONAL_TEST_SET,
            num_examples=test_ex, num_examples_per_label=test_ex_per_label, no_expl=not args.e_pet_test,
            expl_file=args.additional_test_custom_expl_file, concat_expl=args.concat_expl, shuffle_expl=False,
            eval_with_three_labels_explanations_logits=args.eval_with_three_labels_explanations_logits,
            calibration=args.calibration,
            test_gold_expl=args.test_gold_expl)

    if args.no_distillation and not args.lm_training:
        unlabeled_data = None
    else:
        unlabeled_data = load_examples(
            args.task_name, args.data_dir, UNLABELED_SET,
            num_examples=args.unlabeled_examples, no_expl=not args.e_pet_lm,
            expl_file=args.train_custom_expl_file, concat_expl=args.concat_expl, shuffle_expl=args.shuffle_expl_unlabeled)

    if args.calibrate_on_end or args.calibrate_on_step:
        calibration_data = processor.get_calibration_examples()
    else:
        calibration_data = None

    logger.info("Getting metrics ...")
    args.metrics = METRICS.get(args.task_name, DEFAULT_METRICS)

    logger.info("Loading pet configs ...")
    pet_model_cfg, pet_train_cfg, pet_eval_cfg = load_pet_configs(args)
    logger.info("Loading sequence classifier configs ...")
    sc_model_cfg, sc_train_cfg, sc_eval_cfg = load_sequence_classifier_configs(args)
    logger.info("Loading ipet configs ...")
    ipet_cfg = load_ipet_config(args)

    if args.method == 'pet':
        if args.task_name in ['analytical-entailment', 'fantasy-reasoning', 'irony-identification']:
            # train on training data, eval on test data
            pet.train_pet(pet_model_cfg, pet_train_cfg, pet_eval_cfg, sc_model_cfg, sc_train_cfg, sc_eval_cfg,
                        pattern_ids=args.pattern_ids, output_dir=args.output_dir,
                        ensemble_repetitions=args.pet_repetitions, final_repetitions=args.sc_repetitions,
                        reduction=args.reduction, train_data=train_data, unlabeled_data=unlabeled_data,
                        calibration_data=calibration_data,
                        eval_data=test_data, do_train=args.do_train, do_eval=args.do_eval,
                        no_distillation=args.no_distillation, seed=args.seed, 
                        save_train_logits=args.save_train_logits, calibration=args.calibration, eval_type='test',
                        sc_phe=args.sc_phe)
        else:
            # train on training data, eval on dev data
            if not args.no_dev_set:
                pet.train_pet(pet_model_cfg, pet_train_cfg, pet_eval_cfg, sc_model_cfg, sc_train_cfg, sc_eval_cfg,
                            pattern_ids=args.pattern_ids, output_dir=args.output_dir,
                            ensemble_repetitions=args.pet_repetitions, final_repetitions=args.sc_repetitions,
                            reduction=args.reduction, train_data=train_data, unlabeled_data=unlabeled_data,
                            calibration_data=calibration_data,
                            eval_data=dev_data, do_train=args.do_train, do_eval=args.do_eval,
                            no_distillation=args.no_distillation, seed=args.seed, 
                            save_train_logits=args.save_train_logits, calibration=args.calibration, eval_type='dev',
                            sc_phe=args.sc_phe)
            # no training, eval on test data
            pet.train_pet(pet_model_cfg, pet_train_cfg, pet_eval_cfg, sc_model_cfg, sc_train_cfg, sc_eval_cfg,
                        pattern_ids=args.pattern_ids, output_dir=args.output_dir,
                        ensemble_repetitions=args.pet_repetitions, final_repetitions=args.sc_repetitions,
                        reduction=args.reduction, train_data=train_data, unlabeled_data=unlabeled_data,
                        calibration_data=calibration_data,
                        eval_data=test_data, do_train=False, do_eval=args.do_eval,
                        no_distillation=args.no_distillation, seed=args.seed, 
                        save_train_logits=args.save_train_logits, calibration=args.calibration, eval_type='test',
                        sc_phe=args.sc_phe)
            if args.additional_test:
                # no training, eval on additional test data
                pet.train_pet(pet_model_cfg, pet_train_cfg, pet_eval_cfg, sc_model_cfg, sc_train_cfg, sc_eval_cfg,
                            pattern_ids=args.pattern_ids, output_dir=args.output_dir,
                            ensemble_repetitions=args.pet_repetitions, final_repetitions=args.sc_repetitions,
                            reduction=args.reduction, train_data=train_data, unlabeled_data=unlabeled_data,
                            calibration_data=calibration_data,
                            eval_data=additional_test_data, do_train=False, do_eval=args.do_eval,
                            no_distillation=args.no_distillation, seed=args.seed, 
                            save_train_logits=args.save_train_logits, calibration=args.calibration, 
                            eval_type='additional_test', sc_phe=args.sc_phe)

    elif args.method == 'sequence_classifier':
        # baseline: do not use unlabeled data
        sc_train_cfg.use_logits = False 
        sc_train_cfg.lm_training = False 

        if not args.no_dev_set:
            eval_data = dev_data
            eval_type = 'dev'
            pet.train_classifier(sc_model_cfg, sc_train_cfg, sc_eval_cfg, output_dir=args.output_dir,
                                repetitions=args.sc_repetitions, train_data=train_data, unlabeled_data=unlabeled_data,
                                eval_data=eval_data, do_train=args.do_train, do_eval=args.do_eval, seed=args.seed,
                                sc_phe=args.sc_phe, eval_type=eval_type, calibration=args.calibration,
                                save_train_logits=args.save_train_logits,
                                )

        eval_data = test_data
        eval_type = 'test'
        pet.train_classifier(sc_model_cfg, sc_train_cfg, sc_eval_cfg, output_dir=args.output_dir,
                             repetitions=args.sc_repetitions, train_data=train_data, unlabeled_data=unlabeled_data,
                             eval_data=eval_data, do_train=args.no_dev_set, do_eval=args.do_eval, seed=args.seed,
                             sc_phe=args.sc_phe, eval_type=eval_type, calibration=args.calibration,
                             save_train_logits=args.save_train_logits,
                             )

        if args.additional_test:
            eval_data = additional_test_data
            eval_type = 'additional_test'
            pet.train_classifier(sc_model_cfg, sc_train_cfg, sc_eval_cfg, output_dir=args.output_dir,
                repetitions=args.sc_repetitions, train_data=train_data, unlabeled_data=unlabeled_data,
                eval_data=eval_data, do_train=False, do_eval=args.do_eval, seed=args.seed,
                sc_phe=args.sc_phe, eval_type=eval_type
            )

    else:
        raise ValueError(f"Training method '{args.method}' not implemented")


if __name__ == "__main__":
    main()
