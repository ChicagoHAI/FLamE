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
import ast
import json
import os
import random
import statistics
import time
from abc import ABC
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers.data.metrics import simple_accuracy

import log
import wandb
from pet.utils import (
    InputExample,
    exact_match,
    save_logits,
    save_predictions,
    save_labels,
    softmax,
    LogitsList,
    set_seed,
    eq_div,
)
from pet.wrapper import (
    TransformerModelWrapper,
    SEQUENCE_CLASSIFIER_WRAPPER,
    WrapperConfig,
)

logger = log.get_logger("root")


class PetConfig(ABC):
    """Abstract class for a PET configuration that can be saved to and loaded from a json file."""

    def __repr__(self):
        return repr(self.__dict__)

    def save(self, path: str):
        """Save this config to a file."""
        with open(path, "w", encoding="utf8") as fh:
            json.dump(self.__dict__, fh)

    @classmethod
    def load(cls, path: str):
        """Load a config from a file."""
        cfg = cls.__new__(cls)
        with open(path, "r", encoding="utf8") as fh:
            cfg.__dict__ = json.load(fh)
        return cfg


class TrainConfig(PetConfig):
    """Configuration for training a model."""


    def __init__(
        self,
        device: str = None,
        per_gpu_train_batch_size: int = 8,
        per_gpu_unlabeled_batch_size: int = 8,
        n_gpu: int = 1,
        num_train_epochs: int = 3,
        max_steps: int = -1,
        gradient_accumulation_steps: int = 1,
        weight_decay: float = 0.0,
        learning_rate: float = 5e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        max_grad_norm: float = 1,
        lm_training: bool = False,
        use_logits: bool = False,
        alpha: float = 0.9999,
        temperature: float = 1,
        sc_eval_during_train=False, 
        sc_eval_steps=None,
        train_with_all_expl=False,
    ):
        """
        Create a new training config.

        :param device: the device to use ('cpu' or 'gpu')
        :param per_gpu_train_batch_size: the number of labeled training examples per batch and gpu
        :param per_gpu_unlabeled_batch_size: the number of unlabeled examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param num_train_epochs: the number of epochs to train for
        :param max_steps: the maximum number of steps to train for (overrides ``num_train_epochs``)
        :param gradient_accumulation_steps: the number of steps to accumulate gradients for before performing an update
        :param weight_decay: the weight decay to use
        :param learning_rate: the maximum learning rate to use
        :param adam_epsilon: the epsilon value for Adam
        :param warmup_steps: the number of warmup steps to perform before reaching the maximum learning rate
        :param max_grad_norm: the maximum norm for the gradient
        :param lm_training: whether to perform auxiliary language modeling (only for MLMs)
        :param use_logits: whether to use each training example's logits instead of its label (used for distillation)
        :param alpha: the alpha parameter for auxiliary language modeling
        :param temperature: the temperature for distillation
        """
        self.device = device
        self.per_gpu_train_batch_size = per_gpu_train_batch_size
        self.per_gpu_unlabeled_batch_size = per_gpu_unlabeled_batch_size
        self.n_gpu = n_gpu
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.lm_training = lm_training
        self.use_logits = use_logits
        self.alpha = alpha
        self.temperature = temperature
        self.sc_eval_during_train=sc_eval_during_train
        self.sc_eval_steps=sc_eval_steps
        self.train_with_all_expl=train_with_all_expl


class EvalConfig(PetConfig):
    """Configuration for evaluating a model."""

    def __init__(
        self,
        device: str = None,
        n_gpu: int = 1,
        per_gpu_eval_batch_size: int = 8,
        metrics: List[str] = None,
        decoding_strategy: str = "default",
        priming: bool = False,
        eval_result: str = None,
        eval_with_three_labels_explanations_logits: bool = False,
    ):
        """
        Create a new evaluation config.

        :param device: the device to use ('cpu' or 'gpu')
        :param n_gpu: the number of gpus to use
        :param per_gpu_eval_batch_size: the number of evaluation examples per batch and gpu
        :param metrics: the evaluation metrics to use (default: accuracy only)
        :param decoding_strategy: the decoding strategy for PET with multiple masks ('default', 'ltr', or 'parallel')
        :param priming: whether to use priming
        """
        self.device = device
        self.n_gpu = n_gpu
        self.per_gpu_eval_batch_size = per_gpu_eval_batch_size
        self.metrics = metrics
        self.decoding_strategy = decoding_strategy
        self.priming = priming
        self.eval_result = eval_result
        self.eval_with_three_labels_explanations_logits = eval_with_three_labels_explanations_logits


class IPetConfig(PetConfig):
    """Configuration for iterative PET training."""

    def __init__(
        self,
        generations: int = 3,
        logits_percentage: float = 0.25,
        scale_factor: float = 5,
        n_most_likely: int = -1,
    ):
        """
        Create a new iPET config.

        :param generations: the number of generations to train
        :param logits_percentage: the percentage of models to use for annotating training sets for the next generation
        :param scale_factor: the factor by which the training set is increased for each generation
        :param n_most_likely: If >0, in the first generation the n_most_likely examples per label are chosen even
                              if their predicted label is different
        """
        self.generations = generations
        self.logits_percentage = logits_percentage
        self.scale_factor = scale_factor
        self.n_most_likely = n_most_likely


def init_model(config: WrapperConfig) -> TransformerModelWrapper:
    """Initialize a new model from the given config."""
    assert (
        config.pattern_id is not None
    ), "A pattern_id must be set for initializing a new PET model"
    model = TransformerModelWrapper(config)
    return model


def train_ipet(
    ensemble_model_config: WrapperConfig,
    ensemble_train_config: TrainConfig,
    ensemble_eval_config: EvalConfig,
    ipet_config: IPetConfig,
    final_model_config: WrapperConfig,
    final_train_config: TrainConfig,
    final_eval_config: EvalConfig,
    pattern_ids: List[int],
    output_dir: str,
    ensemble_repetitions: int = 3,
    final_repetitions: int = 1,
    reduction: str = "wmean",
    train_data: List[InputExample] = None,
    unlabeled_data: List[InputExample] = None,
    eval_data: List[InputExample] = None,
    do_train: bool = True,
    do_eval: bool = True,
    seed: int = 42,
    sc_phe: bool = False
):
    """
    Train and evaluate a new iPET model for a given task.

    :param ensemble_model_config: the model configuration for each model corresponding to an individual PVP
    :param ensemble_train_config: the training configuration for each model corresponding to an individual PVP
    :param ensemble_eval_config: the evaluation configuration for each model corresponding to an individual PVP
    :param ipet_config: the iPET training configuration
    :param final_model_config: the model configuration for the final distilled sequence classifier
    :param final_train_config: the training configuration for the final distilled sequence classifier
    :param final_eval_config: the evaluation configuration for the final distilled sequence classifier
    :param pattern_ids: the ids of all PVPs to use
    :param output_dir: the output directory
    :param ensemble_repetitions: the number of training repetitions for each model corresponding to an individual PVP
    :param final_repetitions: the number of training repetitions for the final distilled sequence classifier
    :param reduction: the reduction strategy for merging predictions, either 'mean' or 'wmean'
    :param train_data: the training examples to use
    :param unlabeled_data: the unlabeled examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param seed: the random seed to use
    """
    for gen in range(ipet_config.generations):
        gen_output_dir = os.path.join(output_dir, f"g{gen}")

        # Step 1: Train an ensemble of models corresponding to individual patterns
        ipet_data_dir = (
            os.path.join(output_dir, f"g{gen - 1}", "next-gen-train-data")
            if gen > 0
            else None
        )
        train_pet_ensemble(
            ensemble_model_config,
            ensemble_train_config,
            ensemble_eval_config,
            pattern_ids,
            gen_output_dir,
            ipet_data_dir=ipet_data_dir,
            repetitions=ensemble_repetitions,
            train_data=train_data,
            unlabeled_data=unlabeled_data,
            eval_data=eval_data,
            do_train=do_train,
            do_eval=do_eval,
            save_unlabeled_logits=True
        )

        # Step 2: Use the model to annotate examples for the next generation
        original_data_size = (
            len(train_data) if train_data else 10 / ipet_config.scale_factor
        )
        num_new_examples = int(
            original_data_size * (ipet_config.scale_factor ** (gen + 1))
            - len(train_data)
        )
        generate_ipet_train_sets(
            train_data=train_data,
            unlabeled_data=unlabeled_data,
            labels=ensemble_model_config.label_list,
            logits_dir=gen_output_dir,
            output_dir=os.path.join(gen_output_dir, "next-gen-train-data"),
            reduction=reduction,
            num_new_examples=num_new_examples,
            logits_percentage=ipet_config.logits_percentage,
            n_most_likely=ipet_config.n_most_likely if gen == 0 else -1,
            seed=seed,
        )

    # Step 3: Merge the annotations created by each individual model
    logits_dir = os.path.join(output_dir, f"g{ipet_config.generations - 1}")
    logits_file = os.path.join(logits_dir, "unlabeled_logits.txt")
    merge_logits(logits_dir, logits_file, reduction, "unlabeled")
    logits = LogitsList.load(logits_file).logits
    assert len(logits) == len(unlabeled_data)
    logger.info("Got {} logits from file {}".format(len(logits), logits_file))
    for example, example_logits in zip(unlabeled_data, logits):
        example.logits = example_logits

    # Step 4: Train the final sequence classifier model
    final_model_config.wrapper_type = SEQUENCE_CLASSIFIER_WRAPPER
    final_train_config.use_logits = True

    train_classifier(
        final_model_config,
        final_train_config,
        final_eval_config,
        os.path.join(output_dir, "final"),
        repetitions=final_repetitions,
        train_data=train_data,
        unlabeled_data=unlabeled_data,
        eval_data=eval_data,
        do_train=do_train,
        do_eval=do_eval,
        sc_phe=sc_phe
    )


def get_accuracy(
    ensemble_model_config: WrapperConfig,
    ensemble_eval_config: EvalConfig,
    logits,
    eval_data,
    largest_logit='score',
):
    """
    Given eval results, return accuracy
    """
    label_map = {
        label: i for i, label in enumerate(ensemble_model_config.label_list)
    }
    num_labels = len(ensemble_model_config.label_list)

    if ensemble_eval_config.eval_with_three_labels_explanations_logits:
        assert len(logits) % num_labels == 0

        # find largest label logit based on all three (`num_labels`) explanations
        predictions = []
        for example_idx in range(0, len(logits), num_labels):
            if largest_logit == 'score':
                # approach 1: find label for largest logit out of the 9 (num_labels * num_labels) scores
                pred = np.argmax(logits[example_idx: example_idx+num_labels]) % num_labels
            elif largest_logit == 'label_score':
                # approach 2: find largest out of 3 labels (avg across 3 (`num_labels`) explanations)
                pred = np.argmax(np.sum(logits[example_idx: example_idx+num_labels], axis=0))
            else:
                logger.info("invalid arg for largest_logit: {}".format(largest_logit))

            predictions.append(pred)

        predictions = np.array(predictions)

        # get labels: compress eval data labels (3 examples to 1)
        all_labels = [label_map[example.label] for example in eval_data]
        assert len(logits) == len(all_labels)
        labels = []
        for example_idx in range(0, len(all_labels), num_labels):
            label = all_labels[example_idx]
            labels.append(label)

        labels = np.array(labels)
    else: 
        predictions = np.argmax(logits, axis=1)
        # note: eval_data used sequential sampler (although load_examples shuffle the input examples)
        labels = np.array([label_map[example.label] for example in eval_data])

    accuracy = simple_accuracy(predictions, labels)
    return accuracy

def train_pet(
    ensemble_model_config: WrapperConfig,
    ensemble_train_config: TrainConfig,
    ensemble_eval_config: EvalConfig,
    final_model_config: WrapperConfig,
    final_train_config: TrainConfig,
    final_eval_config: EvalConfig,
    pattern_ids: List[int],
    output_dir: str,
    ensemble_repetitions: int = 3,
    final_repetitions: int = 1,
    reduction: str = "wmean",
    train_data: List[InputExample] = None,
    unlabeled_data: List[InputExample] = None,
    eval_data: List[InputExample] = None,
    calibration_data: List[InputExample] = None,
    do_train: bool = True,
    do_eval: bool = True,
    no_distillation: bool = False,
    save_train_logits = False,
    seed: int = 42,
    calibration: bool = False,
    eval_type: str = "dev",
    sc_phe: bool = False
):
    """
    Train and evaluate a new PET model for a given task.

    :param ensemble_model_config: the model configuration for each model corresponding to an individual PVP
    :param ensemble_train_config: the training configuration for each model corresponding to an individual PVP
    :param ensemble_eval_config: the evaluation configuration for each model corresponding to an individual PVP
    :param final_model_config: the model configuration for the final distilled sequence classifier
    :param final_train_config: the training configuration for the final distilled sequence classifier
    :param final_eval_config: the evaluation configuration for the final distilled sequence classifier
    :param pattern_ids: the ids of all PVPs to use
    :param output_dir: the output directory
    :param ensemble_repetitions: the number of training repetitions for each model corresponding to an individual PVP
    :param final_repetitions: the number of training repetitions for the final distilled sequence classifier
    :param reduction: the reduction strategy for merging predictions, either 'mean' or 'wmean'
    :param train_data: the training examples to use
    :param unlabeled_data: the unlabeled examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param no_distillation: if true, no distillation is performed
    :param seed: the random seed to use
    :param calibration: whether to use calibration (ph, phe logits) or not
    """

    # Step 1: Train an ensemble of models corresponding to individual patterns
    logger.info("Train_pet step 1: train an ensemble.")
    logger.info("Time: {}".format(time.ctime()))
    train_pet_ensemble(
        ensemble_model_config,
        ensemble_train_config,
        ensemble_eval_config,
        pattern_ids,
        output_dir,
        repetitions=ensemble_repetitions,
        train_data=train_data,
        unlabeled_data=unlabeled_data,
        calibration_data=calibration_data,
        eval_data=eval_data,
        do_train=do_train,
        do_eval=do_eval,
        save_unlabeled_logits=not no_distillation,
        save_train_logits=save_train_logits,
        seed=seed,
        calibration=calibration,
        eval_type=eval_type
    )

    if save_train_logits:
        logits_file = os.path.join(output_dir, "train_logits.txt")
        merge_logits(output_dir, logits_file, reduction, "train")

    if no_distillation:
        # Evaluate ensemble on eval data:

        # 1. merge annotations created by each individual model
        logits_file = os.path.join(output_dir, "{}_logits.txt".format(eval_type) if not ensemble_eval_config.eval_result else "{}_logits_{}.txt".format(eval_type, ensemble_eval_config.eval_result))
        merge_logits(output_dir, logits_file, reduction, eval_type, 
                     eval_result=ensemble_eval_config.eval_result, 
                     train_with_all_expl=ensemble_train_config.train_with_all_expl)
        logits = LogitsList.load(logits_file).logits
        assert len(logits) == len(eval_data)
        logger.info("Logits len: {}, eval data len: {}".format(len(logits), len(eval_data)))
        logger.info("Got {} logits from file {}".format(len(logits), logits_file))

        # 2. compute accuracy and store it
        # call get_accuracy
        if ensemble_eval_config.eval_with_three_labels_explanations_logits:
            accuracy_score = get_accuracy(
                ensemble_model_config,
                ensemble_eval_config,
                logits,
                eval_data
            )

            results_file = os.path.join(output_dir, "{}_ensemble.txt".format(eval_type) if not ensemble_eval_config.eval_result else "{}_ensemble_{}.txt".format(eval_type, ensemble_eval_config.eval_result))
            print("score accuracy: ", accuracy_score)
            wandb.log({"ensemble-eval-score-accuracy": accuracy_score})
            with open(results_file, "w") as fh:
                fh.write("score accuracy: {} \n".format(accuracy_score))

        else: 
            accuracy = get_accuracy(
                ensemble_model_config,
                ensemble_eval_config,
                logits,
                eval_data,
            )
        
            results_file = os.path.join(output_dir, "{}_ensemble.txt".format(eval_type) if not ensemble_eval_config.eval_result else "{}_ensemble_{}.txt".format(eval_type, ensemble_eval_config.eval_result))
            print("accuracy: ", accuracy)
            wandb.log({"ensemble-eval-accuracy": accuracy})
            with open(results_file, "w") as fh:
                fh.write("{} \n".format(accuracy))

        return

    # Step 2: Merge the annotations created by each individual model
    logger.info("Train_pet step 2: merge annotations.")
    logger.info("Time: {}".format(time.ctime()))
    logits_file = os.path.join(output_dir, "unlabeled_logits.txt")
    merge_logits(output_dir, logits_file, reduction, "unlabeled")
    logits = LogitsList.load(logits_file).logits
    assert len(logits) == len(unlabeled_data)
    logger.info("Got {} logits from file {}".format(len(logits), logits_file))
    for example, example_logits in zip(unlabeled_data, logits):
        example.logits = example_logits

    # Step 3: Train the final sequence classifier model
    logger.info("Train_pet step 3: train final classifier C.")
    logger.info("Time: {}".format(time.ctime()))
    final_model_config.wrapper_type = SEQUENCE_CLASSIFIER_WRAPPER
    final_train_config.use_logits = True

    train_classifier(
        final_model_config,
        final_train_config,
        final_eval_config,
        os.path.join(output_dir, "final"),
        repetitions=final_repetitions,
        train_data=train_data,
        unlabeled_data=unlabeled_data,
        eval_data=eval_data,
        do_train=do_train,
        do_eval=do_eval,
        seed=seed,
        sc_phe=sc_phe,
    )

    logger.info("All done!")
    logger.info("Time: {}".format(time.ctime()))


def train_classifier(
    sc_model_config: WrapperConfig,
    sc_train_config: TrainConfig,
    sc_eval_config: EvalConfig,
    output_dir: str,
    repetitions: int = 3,
    train_data: List[InputExample] = None,
    unlabeled_data: List[InputExample] = None,
    eval_data: List[InputExample] = None,
    do_train: bool = True,
    do_eval: bool = True,
    seed: int = 42,
    sc_phe: bool = False,
    eval_type: str = None,
    calibration: bool = False,
    save_train_logits: bool = False,
):
    """
    Train and evaluate a sequence classification model.

    :param model_config: the model configuration to use
    :param train_config: the training configuration to use
    :param eval_config: the evaluation configuration to use
    :param output_dir: the output directory
    :param repetitions: the number of training repetitions
    :param train_data: the training examples to use
    :param unlabeled_data: the unlabeled examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param seed: the random seed to use
    """

    train_pet_ensemble(
        sc_model_config,
        sc_train_config,
        sc_eval_config,
        pattern_ids=[0],
        output_dir=output_dir,
        repetitions=repetitions,
        train_data=train_data,
        unlabeled_data=unlabeled_data,
        eval_data=eval_data,
        do_train=do_train,
        do_eval=do_eval,
        seed=seed,
        sc_phe=sc_phe,
        eval_type=eval_type,
        calibration=calibration,
        save_train_logits=save_train_logits
    )

    # largest logits with sequence classifier (rather than PET)
    if sc_eval_config.eval_with_three_labels_explanations_logits:
        # load logits
        logits_file = os.path.join(output_dir+"/p0-i0/", "{}_logits.txt".format(eval_type) if not sc_eval_config.eval_result else "{}_logits_{}.txt".format(eval_type, sc_eval_config.eval_result))
        logits = LogitsList.load(logits_file, with_score=False).logits
        # compute accuracy
        accuracy_score = get_accuracy(
            sc_model_config,
            sc_eval_config,
            logits,
            eval_data
        )
        # write result
        results_file = os.path.join(output_dir, "{}_sc.txt".format(eval_type) if not sc_eval_config.eval_result else "{}_sc_{}.txt".format(eval_type, sc_eval_config.eval_result))
        print("score accuracy: ", accuracy_score)
        wandb.log({"sc-eval-score-accuracy": accuracy_score})
        with open(results_file, "w") as fh:
            fh.write("score accuracy: {} \n".format(accuracy_score))


def train_pet_ensemble(
    model_config: WrapperConfig,
    train_config: TrainConfig,
    eval_config: EvalConfig,
    pattern_ids: List[int],
    output_dir: str,
    ipet_data_dir: str = None,
    repetitions: int = 3,
    train_data: List[InputExample] = None,
    unlabeled_data: List[InputExample] = None,
    eval_data: List[InputExample] = None,
    calibration_data: List[InputExample] = None,
    do_train: bool = True,
    do_eval: bool = True,
    save_unlabeled_logits: bool = False,
    save_train_logits: bool = False,
    seed: int = 42,
    calibration: bool = False,
    eval_type: str = "dev",
    sc_phe: bool = False,
):
    """
    Train and evaluate an ensemble of PET models without knowledge distillation.

    :param model_config: the model configuration to use
    :param train_config: the training configuration to use
    :param eval_config: the evaluation configuration to use
    :param pattern_ids: the ids of all PVPs to use
    :param output_dir: the output directory
    :param ipet_data_dir: optional directory containing additional training data for iPET
    :param repetitions: the number of training repetitions
    :param train_data: the training examples to use
    :param unlabeled_data: the unlabeled examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param save_unlabeled_logits: whether logits for unlabeled examples should be saved in a file ``logits.txt``. This
           is required for both iPET and knowledge distillation.
    :param seed: the random seed to use
    :param calibration: whether to use calibration (ph, phe logits) or not
    """

    results = defaultdict(lambda: defaultdict(list))
    set_seed(seed)

    for pattern_id in pattern_ids:
        for iteration in range(repetitions):

            model_config.pattern_id = pattern_id
            results_dict = {}

            pattern_iter_output_dir = "{}/p{}-i{}".format(
                output_dir, pattern_id, iteration
            )

            skipped_training = False
            if os.path.exists(pattern_iter_output_dir):
                logger.warning(
                    f"Path {pattern_iter_output_dir} already exists, skipping the training for it..."
                )
                skipped_training = True
                wrapper = None
            else:
                os.makedirs(pattern_iter_output_dir)

                logger.info("Initialize model ...")
                wrapper = init_model(model_config)

                # Training
                if do_train:
                    if ipet_data_dir:
                        p = os.path.join(
                            ipet_data_dir, "p{}-i{}-train.bin".format(pattern_id, iteration)
                        )
                        ipet_train_data = InputExample.load_examples(p)
                        for example in ipet_train_data:
                            example.logits = None
                    else:
                        ipet_train_data = None

                    logger.info("Train single model ... (one repetition)")
                    logger.info("Time: {}".format(time.ctime()))

                    results_dict.update(
                        train_single_model(
                            wrapper,
                            train_data,
                            calibration_data,
                            train_config,
                            eval_config,
                            ipet_train_data=ipet_train_data,
                            unlabeled_data=unlabeled_data,
                            pattern_iter_output_dir=pattern_iter_output_dir,
                            return_train_set_logits=save_train_logits,
                            eval_data=eval_data,
                            calibration=calibration,
                            sc_phe=sc_phe
                        )
                    )

                    if save_train_logits:
                        logger.info("Saving training logits")
                        save_logits(
                            os.path.join(pattern_iter_output_dir, "train_logits.txt"), results_dict.pop("train_set_logits")
                        )
                    
                    # Saving because the final classifier uses the saved models
                    with open(os.path.join(pattern_iter_output_dir, 'results.txt' if not train_config.train_with_all_expl else  "results_{}.txt".format(eval_config.eval_result)), 'w') as fh:
                        fh.write(str(results_dict))
                    logger.info("Saving trained model at {}...".format(pattern_iter_output_dir))
                    wrapper.save(pattern_iter_output_dir)
                    train_config.save(os.path.join(pattern_iter_output_dir, 'train_config.json'))
                    eval_config.save(os.path.join(pattern_iter_output_dir, '{}_config.json'.format(eval_type)))
                    logger.info("Saving complete")
                    logger.info("Time: {}".format(time.ctime()))

                    if save_unlabeled_logits:
                        start_time = time.time()
                        # do not use explanations for labeling
                        logits = evaluate(
                            wrapper, unlabeled_data, eval_config, no_expl=True, sc_phe=sc_phe
                        )["logits"]
                        save_logits(
                            os.path.join(pattern_iter_output_dir, "logits.txt"), logits
                        )
                        end_time = time.time()
                        logger.info(
                            "Label unlabeled logits took {} seconds".format(
                                round(end_time - start_time, 2)
                            )
                        )

                    if not do_eval:
                        wrapper.model = None
                        wrapper = None
                        torch.cuda.empty_cache()

            # Evaluation
            if do_eval:
                logger.info("Starting evaluation...")
                logger.info("Time: {}".format(time.ctime()))

                logits_file_path = os.path.join(pattern_iter_output_dir, 
                                 "{}_logits.txt".format(eval_type) if not eval_config.eval_result else "{}_logits_{}.txt".format(eval_type, eval_config.eval_result))
                if os.path.exists(logits_file_path):
                    logger.warning(
                        f"Path {logits_file_path} already exists, skipping the evaluation ..."
                    )
                    if wrapper:
                        wrapper.model = None
                        wrapper = None
                    torch.cuda.empty_cache()

                else:

                    if not wrapper:
                        wrapper = TransformerModelWrapper.from_pretrained(
                            pattern_iter_output_dir
                        )

                    eval_result = evaluate(
                        wrapper, eval_data, eval_config, priming_data=train_data, calibration=calibration, sc_phe=sc_phe
                    )

                    save_predictions(
                        os.path.join(pattern_iter_output_dir, 
                                    "{}_predictions.jsonl".format(eval_type) if not eval_config.eval_result else  "{}_predictions_{}.jsonl".format(eval_type, eval_config.eval_result)),
                        wrapper,
                        eval_result,
                    )

                    save_logits(
                        os.path.join(pattern_iter_output_dir, 
                                    "{}_logits.txt".format(eval_type) if not eval_config.eval_result else "{}_logits_{}.txt".format(eval_type, eval_config.eval_result)),
                        eval_result["logits"],
                    )

                    save_labels(
                        os.path.join(pattern_iter_output_dir, 
                                    "{}_true_labels.txt".format(eval_type) if not eval_config.eval_result else "{}_true_labels_{}.txt".format(eval_type, eval_config.eval_result)),
                        wrapper, 
                        eval_result["labels"],
                    )

                    scores = eval_result["scores"]
                    logger.info(
                        "--- RESULT (pattern_id={}, iteration={}) ---".format(
                            pattern_id, iteration
                        )
                    )
                    logger.info(scores)
                    logger.info("Time: {}".format(time.ctime()))

                    results_dict["test_set_after_training"] = scores
                    with open(
                        os.path.join(pattern_iter_output_dir, "{}_results.json".format(eval_type) if not eval_config.eval_result else "{}_results_{}.json".format(eval_type, eval_config.eval_result)), "w"
                    ) as fh:
                        json.dump(results_dict, fh)

                    wandb.log(
                        {"p{}-i{}-results".format(pattern_id, iteration): results_dict}
                    )

                    for metric, value in scores.items():
                        results[metric][pattern_id].append(value)

                    wrapper.model = None
                    wrapper = None
                    torch.cuda.empty_cache()

    if do_eval:
        logger.info("=== OVERALL RESULTS ===")
        if len(results.keys()) != 0:
            _write_results(os.path.join(output_dir, "{}_result.txt".format(eval_type) if not eval_config.eval_result else "{}_result_{}.txt".format(eval_type, eval_config.eval_result)), results)
    else:
        logger.info("=== ENSEMBLE TRAINING COMPLETE ===")


def train_single_model(
    model: TransformerModelWrapper,
    train_data: List[InputExample],
    calibration_data: List[InputExample],
    config: TrainConfig,
    eval_config: EvalConfig = None,
    ipet_train_data: List[InputExample] = None,
    unlabeled_data: List[InputExample] = None,
    return_train_set_results: bool = True,
    pattern_iter_output_dir: str = None,
    return_train_set_logits: bool = False,
    eval_data=None,
    calibration=False,
    sc_phe=False
):
    """
    Train a single model.

    :param model: the model to train
    :param train_data: the training examples to use
    :param config: the training config
    :param eval_config: the evaluation config
    :param ipet_train_data: an optional list of iPET training examples to use
    :param unlabeled_data: an optional list of unlabeled examples to use
    :param return_train_set_results: whether results on the train set before and after training should be computed and
           returned
    :return: a dictionary containing the global step, average loss and (optionally) results on the train set
    """

    device = torch.device(
        config.device
        if config.device
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    if not ipet_train_data:
        ipet_train_data = []

    results_dict = {}

    logger.info("device: {}".format(device))
    model.model.to(device)

    if train_data and return_train_set_results:
        logger.info("Evaluate on training set before training ...")
        results_dict["train_set_before_training"] = evaluate(
            model, train_data, eval_config, sc_phe=sc_phe
        )["scores"]["acc"]

    all_train_data = train_data + ipet_train_data

    if not all_train_data and not config.use_logits:
        logger.warning("Training method was called without training examples")
    else:
        global_step, tr_loss = model.train(
            all_train_data,
            calibration_data,
            device,
            per_gpu_train_batch_size=config.per_gpu_train_batch_size,
            per_gpu_unlabeled_batch_size=config.per_gpu_unlabeled_batch_size,
            n_gpu=config.n_gpu,
            num_train_epochs=config.num_train_epochs,
            max_steps=config.max_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            adam_epsilon=config.adam_epsilon,
            warmup_steps=config.warmup_steps,
            max_grad_norm=config.max_grad_norm,
            unlabeled_data=unlabeled_data
            if config.lm_training or config.use_logits
            else None,
            lm_training=config.lm_training,
            use_logits=config.use_logits,
            alpha=config.alpha,
            temperature=config.temperature,
            pattern_iter_output_dir=pattern_iter_output_dir,
            train_config=config,
            eval_config=eval_config,
            sc_eval_during_train=config.sc_eval_during_train,
            sc_eval_steps=config.sc_eval_steps,
            eval_data=eval_data,
            calibration=calibration,
            sc_phe=sc_phe
        )
        results_dict["global_step"] = global_step
        results_dict["average_loss"] = tr_loss

    if train_data and (return_train_set_results or return_train_set_logits):
        train_eval_results = evaluate(
            model, train_data, eval_config, sc_phe=sc_phe
        )
        if return_train_set_results:
            results_dict["train_set_after_training"] = train_eval_results["scores"]["acc"]
        
        if return_train_set_logits:
            results_dict["train_set_logits"] = train_eval_results["logits"]

    return results_dict


def evaluate(
    model: TransformerModelWrapper,
    eval_data: List[InputExample],
    config: EvalConfig,
    priming_data: List[InputExample] = None,
    no_expl: bool = False,
    calibration: bool = False,
    sc_phe: bool = False
) -> Dict:
    """
    Evaluate a model.

    :param model: the model to evaluate
    :param eval_data: the examples for evaluation
    :param config: the evaluation config
    :param priming_data: an optional list of priming data to use
    :return: a dictionary containing the model's logits, predictions and (if any metrics are given) scores
    """

    if config.priming:
        for example in eval_data:
            example.meta["priming_data"] = priming_data

    metrics = config.metrics if config.metrics else ["acc"]
    device = torch.device(
        config.device
        if config.device
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.model.to(device)
    
    results = model.eval(
        eval_data,
        device,
        per_gpu_eval_batch_size=config.per_gpu_eval_batch_size,
        n_gpu=config.n_gpu,
        decoding_strategy=config.decoding_strategy,
        priming=config.priming,
        no_expl=no_expl,
        calibration=calibration,
        sc_phe=sc_phe
    )
    
    predictions = np.argmax(results["logits"], axis=1)
    scores = {}

    for metric in metrics:
        if metric == "acc":
            scores[metric] = simple_accuracy(predictions, results["labels"])
        elif metric == "f1":
            scores[metric] = f1_score(results["labels"], predictions)
        elif metric == "f1-macro":
            scores[metric] = f1_score(results["labels"], predictions, average="macro")
        elif metric == "em":
            scores[metric] = exact_match(
                predictions, results["labels"], results["question_ids"]
            )
        else:
            raise ValueError(f"Metric '{metric}' not implemented")

    results["scores"] = scores
    results["predictions"] = predictions
    return results


def _write_results(path: str, results: Dict):
    with open(path, "w") as fh:
        for metric in results.keys():
            for pattern_id, values in results[metric].items():
                mean = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0
                result_str = "{}-p{}: {} +- {}".format(metric, pattern_id, mean, stdev)
                logger.info(result_str)
                fh.write(result_str + "\n")

        for metric in results.keys():
            all_results = [
                result
                for pattern_results in results[metric].values()
                for result in pattern_results
            ]
            all_mean = statistics.mean(all_results)
            all_stdev = statistics.stdev(all_results) if len(all_results) > 1 else 0
            result_str = "{}-all-p: {} +- {}".format(metric, all_mean, all_stdev)
            logger.info(result_str)
            fh.write(result_str + "\n")


def merge_logits(logits_dir: str, output_file: str, reduction: str, data_type: str, 
                 eval_result: str=None, train_with_all_expl: bool=False):
    """
    Merge the logits predicted by multiple models that are stored in the subdirectories.

    :param logits_dir: a directory for which each sub-directory corresponds to a pretrained model and contains
           both a file ``results.txt`` containing that model's results on the training set and a file ``logits.txt``
           containing that model's predictions for the unlabeled data.
    :param output_file: the file to which the merged logits for all unlabeled examples are written.
    :param reduction: the strategy for merging logits, either 'mean' or 'wmean'. For 'mean', all models contribute
           equally, for 'wmean', each model's contribution is proportional to its accuracy on the training set before
           training.
    :param data_type: 'eval' or 'unlabeled'.
    """
    subdirs = next(os.walk(logits_dir))[1]
    logger.info(
        "Found the following {} subdirectories: {}".format(len(subdirs), subdirs)
    )

    all_logits_lists = []

    for subdir in subdirs:
        results_file = os.path.join(logits_dir, subdir, "results.txt" if not train_with_all_expl else "results_{}.txt".format(eval_result))
        assert data_type in ["dev", "test", "unlabeled", "train", "additional_test"]
        if data_type == "dev":
            logits_file = os.path.join(logits_dir, subdir, "dev_logits.txt" if not eval_result else "dev_logits_{}.txt".format(eval_result))
        elif data_type == "test":
            logits_file = os.path.join(logits_dir, subdir, "test_logits.txt" if not eval_result else "test_logits_{}.txt".format(eval_result))
        elif data_type == "unlabeled":
            logits_file = os.path.join(logits_dir, subdir, "logits.txt")
        elif data_type == "train":
            logits_file = os.path.join(logits_dir, subdir, "train_logits.txt")
        elif data_type == "additional_test":
            logits_file = os.path.join(logits_dir, subdir, "additional_test_logits.txt" if not eval_result else "additional_test_logits_{}.txt".format(eval_result))
        logits = []

        if not os.path.exists(logits_file):
            logger.warning(
                f"Skipping subdir '{subdir}' because 'logits.txt' not found"
            )
            continue

        if reduction == "mean":
            result_train = 1
        else:
            if not os.path.exists(results_file):
                logger.warning(
                    f"Skipping subdir '{subdir}' because 'results.txt' not found"
                )
                continue
            with open(results_file, "r") as fh:
                results = ast.literal_eval(fh.read())
                result_train = results["train_set_before_training"]

        with open(logits_file, "r") as fh:
            for line in fh.read().splitlines():
                example_logits = [float(x) for x in line.split()]
                logits.append(example_logits)

        logger.info(
            "File {}: Score = {}, #Logits = {}, #Labels = {}".format(
                results_file, result_train, len(logits), len(logits[0])
            )
        )

        loglist = LogitsList(score=result_train, logits=logits)
        all_logits_lists.append(loglist)

    merged_loglist = merge_logits_lists(all_logits_lists, reduction=reduction)
    merged_loglist.save(output_file)


def merge_logits_lists(
    logits_lists: List[LogitsList], reduction: str = "mean"
) -> LogitsList:
    """
    Merge a list of :class:`LogitsList` objects.

    :param logits_lists: the lists to merge
    :param reduction: the strategy for merging logits, either 'mean' or 'wmean'. For 'mean', all models contribute
           equally, for 'wmean', each model's contribution is proportional to its accuracy on the training set before
           training.
    :return: the merged list
    """
    assert len(set(len(ll.logits) for ll in logits_lists)) == 1
    logits = np.array([ll.logits for ll in logits_lists])
    weights = np.array([ll.score for ll in logits_lists])

    if reduction == "mean":
        logits = np.mean(logits, axis=0).tolist()
    elif reduction == "wmean":
        logits = np.average(logits, axis=0, weights=weights).tolist()
    else:
        raise ValueError("Reduction strategy '{}' not implemented".format(reduction))

    return LogitsList(score=-1, logits=logits)


def generate_ipet_train_sets(
    train_data: List[InputExample],
    unlabeled_data: List[InputExample],
    labels: List[str],
    logits_dir: str,
    output_dir: str,
    reduction: str,
    num_new_examples: int,
    logits_percentage: float,
    n_most_likely: int = -1,
    seed: int = 42,
):
    """
    Generate training sets for the next generation of iPET models.

    :param train_data: the training examples
    :param unlabeled_data: the unlabeled examples
    :param labels: the list of all possible labels
    :param logits_dir: the directory that contains the predictions of all models in the current generation for the
           unlabeled data.
    :param output_dir: the output directory
    :param reduction: the strategy for merging logits, either 'mean' or 'wmean'. For 'mean', all models contribute
           equally, for 'wmean', each model's contribution is proportional to its accuracy on the training set before
           training.
    :param num_new_examples: the number of new examples to create
    :param logits_percentage: the percentage of models to use for annotating training sets for the next generation
    :param n_most_likely: If >0, in the first generation the n_most_likely examples per label are chosen even
                              if their predicted label is different
    :param seed: the random seed to use
    """
    subdirs = next(os.walk(logits_dir))[1]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info(
        "Found the following {} subdirectories: {}".format(len(subdirs), subdirs)
    )

    if train_data:
        train_examples_per_label = [
            sum(1 for ex in train_data if ex.label == label) for label in labels
        ]
        multiplier = num_new_examples / len(train_data)
        examples_per_label = [int(epl * multiplier) for epl in train_examples_per_label]
        logger.info(
            f"Example distribution in the original dataset: {train_examples_per_label}"
        )
    else:
        examples_per_label = eq_div(num_new_examples, len(labels))

    logger.info(f"Target distribution for the new dataset: {examples_per_label}")

    for example in unlabeled_data:
        example.label, example.logits = None, None

    logits_lists = {}

    rng = random.Random(seed)
    rng_np = np.random.RandomState(seed)

    for subdir in subdirs:
        results_file = os.path.join(logits_dir, subdir, "results.txt")
        logits_file = os.path.join(logits_dir, subdir, "logits.txt")
        logits = []

        if not os.path.exists(results_file) or not os.path.exists(logits_file):
            logger.warning(
                f"Skipping subdir '{subdir}' because 'results.txt' or 'logits.txt' not found"
            )
            continue

        if reduction == "mean":
            result_train = 1
        else:
            with open(results_file, "r") as fh:
                results = ast.literal_eval(fh.read())
                result_train = results["train_set_before_training"]

        with open(logits_file, "r") as fh:
            for line in fh.read().splitlines():
                example_logits = [float(x) for x in line.split()]
                logits.append(example_logits)

        logger.info(
            "File {}: Score = {}, #Logits = {}, #Labels = {}".format(
                results_file, result_train, len(logits), len(logits[0])
            )
        )

        loglist = LogitsList(score=result_train, logits=logits)
        logits_lists[subdir] = loglist

    for subdir in subdirs:
        other_logits_lists = [ll for sd, ll in logits_lists.items() if sd != subdir]
        subdir_train_set = generate_ipet_train_set(
            other_logits_lists,
            labels=labels,
            original_data=unlabeled_data,
            examples_per_label=examples_per_label,
            logits_percentage=logits_percentage,
            reduction=reduction,
            n_most_likely=n_most_likely,
            rng=rng,
            rng_np=rng_np,
        )

        InputExample.save_examples(
            subdir_train_set, os.path.join(output_dir, subdir + "-train.bin")
        )


def generate_ipet_train_set(
    logits_lists: List[LogitsList],
    labels: List[str],
    original_data: List[InputExample],
    examples_per_label: List[int],
    logits_percentage: float,
    reduction: str = "mean",
    n_most_likely: int = -1,
    rng=None,
    rng_np=None,
) -> List[InputExample]:
    """
    Generate a single training set for the next generation of iPET models.

    :param logits_lists: predictions from the previous generation of models
    :param labels: all task labels
    :param original_data: the original training data corresponding to the logits_lists
    :param examples_per_label: the number of examples per label to create
    :param logits_percentage: the percentage of models/logits to choose
    :param reduction: the reduction strategy ('wmean' or 'mean')
    :param n_most_likely: if >0, for each label the n_most_likely examples with the highest logits are chosen
    :param rng: the random number generator to use for non-numpy operations
    :param rng_np: the random number generator to use for numpy operations
    :return: a list of input examples that serves as training set for the next generation
    """

    assert len(set(len(ll.logits) for ll in logits_lists)) == 1

    if not rng:
        rng = random.Random()
    if not rng_np:
        rng_np = np.random.RandomState()

    num_logits_lists = round(len(logits_lists) * logits_percentage)
    logits_lists = rng.sample(logits_lists, k=num_logits_lists)
    logits = np.array([ll.logits for ll in logits_lists])
    weights = np.array([ll.score for ll in logits_lists])

    if reduction == "mean":
        logits = np.mean(logits, axis=0)
        logits = softmax(logits, axis=1).tolist()
    elif reduction == "wmean":
        logits = np.average(logits, axis=0, weights=weights)
        logits = softmax(logits, axis=1).tolist()
    else:
        raise ValueError("Reduction strategy '{}' not implemented".format(reduction))

    assert len(logits) == len(original_data)

    for lgs, example in zip(logits, original_data):
        example.logits = lgs
        example.label = labels[np.argmax(example.logits).item()]

    test_set = []

    for idx, label in enumerate(labels):

        if n_most_likely <= 0:
            examples = [ex for ex in original_data if ex.label == label]
            logger.info(
                "There are {} examples for label {}".format(len(examples), label)
            )
            while len(examples) < examples_per_label[idx]:
                # upsample examples if there are too few
                examples.extend(ex for ex in original_data if ex.label == label)
        else:
            examples = [
                (ex.logits[idx], ex_idx, ex) for ex_idx, ex in enumerate(original_data)
            ]
            examples.sort(reverse=True)
            examples = [ex for score, ex_idx, ex in examples[:n_most_likely]]
            examples = [deepcopy(ex) for ex in examples]
            for example in examples:
                example.logits = [example.logits[idx]]
                example.label = label

        label_examples = _draw_examples_by_label_probability(
            examples=examples, num_examples=examples_per_label[idx], rng=rng_np
        )
        test_set.extend(label_examples)

    return test_set


def _draw_examples_by_label_probability(
    examples: List[InputExample], num_examples: int, rng
) -> List[InputExample]:
    label_probabilities = [max(example.logits) for example in examples]
    sum_label_probabilities = sum(label_probabilities)
    label_probabilities = [p / sum_label_probabilities for p in label_probabilities]
    return rng.choice(
        examples, size=num_examples, replace=False, p=label_probabilities
    ).tolist()

