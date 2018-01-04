import random

from tensorflow.examples.tutorials.mnist import input_data

WIDTH = 28
HEIGHT = 28

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

import tensorflow as tf
import numpy as np

################################
# Enter your code between here #
################################
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    feature_columns = [tf.feature_column.numeric_column("x", shape=[WIDTH * HEIGHT])]
    tf.reset_default_graph()
    classif = tf.estimator.DNNClassifier(hidden_units=[10, 10, 10], feature_columns=feature_columns, n_classes=10,
                                         optimizer=tf.train.GradientDescentOptimizer(
                                             learning_rate=0.002,
                                         )
                                         )
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

    train_input_fn = tf.estimator.inputs.numpy_input_fn({"x": mnist.train.images}, train_labels, batch_size=200,
                                                        num_epochs=None, shuffle=True)
    classif.train(input_fn=train_input_fn, steps=2000)
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": mnist.test.images},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    accuracy_score = classif.evaluate(input_fn=test_input_fn)["accuracy"]
    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

########################
# predict_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={"x": mnist.test.images},
#     num_epochs=1,
#     shuffle=False)
# predictions = list(classif.predict(input_fn=predict_input_fn))
# predicted_classes = [p["classes"][0].decode() for p in predictions]
# print(' '.join(map(str, predicted_classes)))
#        And here      #

# Uncomment to get a prediction number for each image

# result = sess.run(tf.argmax(y, 1), feed_dict={x: mnist.validation.images})
# print(' '.join(map(str, result)))
