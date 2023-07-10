# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
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

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import os
import deepspeed as ds
import datasets
import sde_lib
import deepspeed_losses as losses
from absl import flags
from models import ncsnpp
from models import utils as mutils
from models.ema import ExponentialMovingAverage

FLAGS = flags.FLAGS


def train(config, workdir):
    """Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """

    # Initialize dataset
    train_dataset, _ = datasets.get_dataset(config)
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)

    # Initialize DeepSpeed engine
    model_name = config.model.name
    score_model = mutils.get_model(model_name)(config)
    model_engine, _, train_loader, _ = ds.initialize(
        args=config,
        model=score_model,
        training_data=train_dataset,
    )

    ema = ExponentialMovingAverage(
        model_engine.parameters(), decay=config.model.ema_rate
    )
    state = dict(model_engine=model_engine, ema=ema)

    # Create checkpoints directory and save initial checkpoint
    checkpoint_dir = os.path.join(workdir, "checkpoints", config.data.dataset)
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_engine.save_checkpoint(
        checkpoint_dir, tag=f"step_{model_engine.global_steps}"
    )

    # Setup SDEs
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
    elif config.training.sde.lower() == "subvpsde":
        sde = sde_lib.subVPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
    elif config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(
            sigma_min=config.model.sigma_min,
            sigma_max=config.model.sigma_max,
            N=config.model.num_scales,
        )
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(
        sde,
        train=True,
        reduce_mean=reduce_mean,
        continuous=continuous,
        likelihood_weighting=likelihood_weighting,
    )

    num_train_steps = config.training.n_iters
    while model_engine.global_steps < num_train_steps:
        for batch in train_loader:
            if isinstance(batch, list):
                batch, _ = batch
            batch = batch.to(model_engine.device)
            batch = scaler(batch)
            # Execute one training step
            train_step_fn(state, batch)

            # Save a checkpoint periodically and generate samples if needed
            if (
                model_engine.global_steps != 0
                and model_engine.global_steps % config.training.snapshot_freq == 0
                or model_engine.global_steps == num_train_steps
            ):
                # Save the checkpoint
                model_engine.save_checkpoint(
                    checkpoint_dir, tag=f"step_{model_engine.global_steps}"
                )
