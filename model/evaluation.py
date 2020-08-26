"""Tensorflow utility functions for testing"""
import os
import datetime
import time

import tensorflow as tf
from tensorflow import keras
import numpy as np

class Evaluate():
    
    def __init__(self, model_spec, test_ds):

        self.model_spec = model_spec
        self.test_ds =  test_ds
        # Get relevant graph operations or nodes needed for training
        self.model = self.model_spec

        self.accuracy_rmse = tf.keras.metrics.RootMeanSquaredError(name='train_rmse')
        self.accuracy_mse = tf.keras.metrics.MeanSquaredError(name='train_mse')
        self.accuracy_mae = tf.keras.metrics.MeanAbsoluteError(name='train_mae')

    @tf.function
    def test_step(self, x_test, y_test):
        """Testing the model on batches
        Args:
            model_spec: contains the graph operations or nodes needed for training
            x_train: training images
            y_train: training measured forces
        """
        y_test_pred = self.model(x_test, training=False)
        # write metices to writer for summary use
        self.accuracy_rmse.update_state(y_test, y_test_pred)
        self.accuracy_mse.update_state(y_test, y_test_pred)
        self.accuracy_mae.update_state(y_test, y_test_pred)

    def test(self, params):
        """Train the model and evaluate every epoch.
        Args:
        train_model_spec: (dict) contains the graph operations or nodes needed for training
        params: (Params) contains hyperparameters of the model.
            Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        test_ds: training dataset
        """
        print("[INFO] testing started ...")
        # loop over the number of epochs
        epochStart = time.time()

        # Compute number of batches in one epoch (one full pass over the training set)
        num_steps = int(np.ceil(params.test_size / params.batch_size))  
        # ----------------------------------------------------------------------
        for x_test, y_test in self.test_ds.take(num_steps):
            self.test_step(x_test, y_test)
            # Display metrics at the end of each epoch.
            metrics = {
                        "Test_RMSE": '{:04.2f}'.format(self.accuracy_rmse.result().numpy()),
                        "Test_MSE": '{:04.2f}'.format(self.accuracy_mse.result().numpy()),
                        "Test_MAE": '{:04.2f}'.format(self.accuracy_mae.result().numpy()),
                        }      
        # ----------------------------------------------------------------------
        # end of test
        self.accuracy_rmse.reset_states()
        self.accuracy_mse.reset_states()
        self.accuracy_mae.reset_states()

        epochEnd = time.time()
        elapsed = (epochEnd - epochStart) / 60.0

        # show timing information for the epoch
        template = '[INFO] Test_RMSE: {}, Test_MSE: {}, Test_MAE: {}'
        print(template.format(metrics["Test_RMSE"], metrics["Test_MSE"], metrics["Test_MAE"]))

        print("[INFO] Took {:.4} minutes".format(elapsed))
        print("[INFO] Testing done")