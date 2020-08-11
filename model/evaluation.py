"""Tensorflow utility functions for testing"""

# @tf.function
def test_step(model_spec, x_test, y_test, params):
    """Testing the model on batches
    Args:
        model_spec: (dict) contains the graph operations or nodes needed for training
        params: (Params) hyperparameters
    """

    # Get relevant graph operations or nodes needed for training
    model = model_spec['model']
    loss_object = model_spec['loss']
    metrics = model_spec['metrics']
    opt = model_spec['opt']
    test_loss = metrics['test_loss']
    test_accuracy = metrics['test_accuracy']

    # keep track of our gradients
    with tf.GradientTape() as tape:
        # make a prediction using the model and then calculate the loss
        y_test_pred = model(x_test)
        loss = loss_object(y_test, y_test_pred)
    # write metices to writer for summary use
    test_los = test_loss(loss)
    test_acc = test_accuracy(y_test, y_test_pred)

    return test_los, test_acc