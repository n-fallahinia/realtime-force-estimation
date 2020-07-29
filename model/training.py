"""Tensorflow utility functions for training"""

import os
import datetime
import time

from tqdm import trange
import tensorflow as tf
import numpy as np


def train_step(model_spec, x_train, y_train, params):
    """Train the model on batches
    Args:
        model_spec: (dict) contains the graph operations or nodes needed for training
        writer: (tf.summary.FileWriter) writer for summaries
        params: (Params) hyperparameters
    """

    # Get relevant graph operations or nodes needed for training
    model = model_spec['model']
    loss_object = model_spec['loss']
    metrics = model_spec['metrics']

    train_loss = metrics['train_loss']
    train_accuracy = metrics['train_accuracy']

    # keep track of our gradients
    with tf.GradientTape() as tape:
        # make a prediction using the model and then calculate the loss
        y_train_pred = model(x_train, training=True)
        loss = loss_object(y_train, y_train_pred)
    # calculate the gradients using our tape and then update the model weights
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    # write metices to writer for summary use
    train_loss(loss)
    train_accuracy(y_train, y_train_pred)


def training_and_eval(train_model_spec, log_dir, params, train_ds = None, restore_from=None):
    """Train the model and evaluate every epoch.
    Args:
        train_model_spec: (dict) contains the graph operations or nodes needed for training
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        train_ds: training dataset
        log_dir: directory for log
        restore_from: (string) directory or file containing weights to restore the graph
    """
    # set up the train summary writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(log_dir, current_time , 'train_summaries')
    eval_log_dir = os.path.join(log_dir, current_time , 'eval_summaries')

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)

    begin_at_epoch = 0

    # !!!!!!!!!!!!!!!!!!!!!!!!!!
    # Reload weights from directory if specified
    if restore_from is not None:
        print("Restoring parameters from {}".format(restore_from))
        if os.path.isdir(restore_from):
            pass
        # TODO  model restore from checkpoint
    # !!!!!!!!!!!!!!!!!!!!!!!!!!

    # loop over the number of epochs
    for epoch in range(begin_at_epoch, begin_at_epoch + params.num_epochs):
        # show the current epoch number
        print("[INFO] starting epoch {}/{}...".format(epoch + 1, params.num_epochs))
        # sys.stdout.flush()
        epochStart = time.time()
        # Compute number of batches in one epoch (one full pass over the training set)
        num_steps = int(np.ceil(params.train_size / params.batch_size)) 

        # Use tqdm for progress bar
        t = trange(num_steps)
        # loop over the data in batch size increments
        for i in t :
            # determine starting and ending slice indexes for the current batch
            start = i * params.batch_size
            end = start + params.batch_size
            # # TODO train_step(train_model_spec, x_train[start:end], y_train[start:end] ,params)
            pass
        # show timing information for the epoch
        epochEnd = time.time()
        elapsed = (epochEnd - epochStart) / 60.0
        # Log the loss in the tqdm progress bar

        # !!!!!!!!!!!!!!!!!!!!!!!!!!
        # TODO  eval session
        # !!!!!!!!!!!!!!!!!!!!!!!!!!

        # !!!!!!!!!!!!!!!!!!!!!!!!!!
        # TODO  save the best eval
        # !!!!!!!!!!!!!!!!!!!!!!!!!!
