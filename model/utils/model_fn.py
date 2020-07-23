"""Define the model."""

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential


def buil_model(is_training, width, height, depth, classes):
    
    """Compute logits of the model (output distribution)
    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters
    Returns:
        output: output of the model
    """
    IMG_SHAPE = (height, width, depth)
    chanDim = -1
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
    # Show a summary of the model. Check the number of trainable parameters
    base_model.trainable = False
    # global average layer (also faltten the output of MobileNet)
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    # first FC layer with droupout
    linear_layer = tf.keras.layers.Dense(1024, activation='relu')
    dropout_layer = tf.keras.layers.Dropout(0.5)
    # prediction layer with 3 outputs
    prediction_layer = tf.keras.layers.Dense(classes, activation= 'linear')
    # build the model using Keras' Sequential API
    model = Sequential([
        # MobileNet => avgPOOL => FC => RELU => DROPOUT => FC => LINEAR
        base_model,
        global_average_layer,
        linear_layer,
        dropout_layer,
        prediction_layer
    ])
    
    # return the constructed network architecture

    EPOCHS = 5
    BS = 32
    INIT_LR = 1e-3

    loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    opt = tf.keras.optimizers.RMSprop(lr=INIT_LR, momentum=0.9)

    train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')
    train_accuracy = tf.keras.metrics.RootMeanSquaredError(name='train_accuracy')
    
    return model