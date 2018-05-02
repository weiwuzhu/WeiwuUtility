#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_float("test_sample_percentage", .2, "Percentage of the test data")
tf.flags.DEFINE_string("training_data_file", "./data/YelpClick/YelpClickTrainingData.tsv", "Data source for the training data.")
tf.flags.DEFINE_string("class_index_file", "./data/YelpClick/YelpClickCategoryIndex.tsv", "Output file for the class index.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1525138392/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on test data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.training_data_file, FLAGS.class_index_file, True)
    test_sample_index = -1 * int(FLAGS.test_sample_percentage * float(len(y_test)))
    x_raw, y_test = x_raw[test_sample_index:], y_test[test_sample_index:]
else:
    x_raw = [data_helpers.clean_str("Auto Repair;Oil Change Stations;Transmission Repair")]
    y_test = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    y_test[0][10] = 1

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = np.array([], dtype=np.float).reshape(0,len(y_test[0]))

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
correct_predictions = []
if y_test is not None:
    correct_predictions = [1 if (i==j).all() else 0 for i, j in zip(all_predictions, y_test)]
    correct_predictions_count = float(sum(correct_predictions))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions_count/float(len(y_test))))

# Save the evaluation to a csv
truth_str = data_helpers.label2string(y_test, FLAGS.class_index_file)
prediction_str = data_helpers.label2string(all_predictions, FLAGS.class_index_file)
if not FLAGS.eval_train:
    print("Predicted classes of the first one are: " + prediction_str[0])
predictions_human_readable = np.column_stack((np.array(x_raw), prediction_str, truth_str))
if y_test is not None:
    predictions_human_readable = np.column_stack((np.array(x_raw), prediction_str, truth_str, correct_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.txt")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    f.write('input\tprediction\tlabel\tcorrect\n')
    for i in predictions_human_readable:
        f.write('\t'.join(i) + '\n')
