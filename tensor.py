#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 16:55:01 2016
@author: jonmcewan
"""
#adapted from: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/learn/wide_n_deep_tutorial.py
import tensorflow as tf
import pandas as pd
import tempfile
import os


#functions to build the model
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "", "Base directory for output models.")
flags.DEFINE_string("model_type", "wide_n_deep",
                    "Valid model types: {'wide', 'deep', 'wide_n_deep'}.")
flags.DEFINE_integer("train_steps", 200, "Number of training steps.")
flags.DEFINE_string(
    "train_data",
    "",
    "Path to the training data.")
flags.DEFINE_string(
    "test_data",
    "",
"Path to the test data.")   



train = FLAGS.train_data
test=FLAGS.test_data

COLUMNS = ["id","cat1","cat2","cat3","cat4","cat5","cat6","cat7","cat8","cat9","cat10",
"cat11","cat12","cat13","cat14","cat15","cat16","cat17","cat18","cat19","cat20","cat21","cat22",
"cat23","cat24","cat25","cat26","cat27","cat28","cat29","cat30","cat31","cat32","cat33","cat34",
"cat35","cat36","cat37","cat38","cat39","cat40","cat41","cat42","cat43","cat44","cat45","cat46",
"cat47","cat48","cat49","cat50","cat51","cat52","cat53","cat54","cat55","cat56","cat57","cat58",
"cat59","cat60","cat61","cat62","cat63","cat64","cat65","cat66","cat67","cat68","cat69","cat70",
"cat71","cat72","cat73","cat74","cat75","cat76","cat77","cat78","cat79","cat80","cat81","cat82",
"cat83","cat84","cat85","cat86","cat87","cat88","cat89","cat90","cat91","cat92","cat93","cat94",
"cat95","cat96","cat97","cat98","cat99","cat100","cat101","cat102","cat103","cat104","cat105",
"cat106","cat107","cat108","cat109","cat110","cat111","cat112","cat113","cat114","cat115","cat116",
"cont1","cont2","cont3","cont4","cont5","cont6","cont7","cont8","cont9","cont10","cont11","cont12","cont13","cont14","loss"
]
LABEL_COLUMN = "loss"
CATEGORICAL_COLUMNS = ["cat1","cat2","cat3","cat4","cat5","cat6","cat7","cat8","cat9","cat10",
"cat11","cat12","cat13","cat14","cat15","cat16","cat17","cat18","cat19","cat20","cat21","cat22",
"cat23","cat24","cat25","cat26","cat27","cat28","cat29","cat30","cat31","cat32","cat33","cat34",
"cat35","cat36","cat37","cat38","cat39","cat40","cat41","cat42","cat43","cat44","cat45","cat46",
"cat47","cat48","cat49","cat50","cat51","cat52","cat53","cat54","cat55","cat56","cat57","cat58",
"cat59","cat60","cat61","cat62","cat63","cat64","cat65","cat66","cat67","cat68","cat69","cat70",
"cat71","cat72","cat73","cat74","cat75","cat76","cat77","cat78","cat79","cat80","cat81","cat82",
"cat83","cat84","cat85","cat86","cat87","cat88","cat89","cat90","cat91","cat92","cat93","cat94",
"cat95","cat96","cat97","cat98","cat99","cat100","cat101","cat102","cat103","cat104","cat105",
"cat106","cat107","cat108","cat109","cat110","cat111","cat112","cat113","cat114","cat115","cat116"]
CONTINUOUS_COLUMNS = ["cont1","cont2","cont3","cont4","cont5","cont6","cont7","cont8","cont9","cont10","cont11","cont12","cont13","cont14"]

df_train = pd.read_csv(train)
df_test = pd.read_csv(test)
     
    
def build_estimator(model_dir):
  """Build an estimator."""

  # Continuous base columns.
  continuous_cols = []
  continuous_len = len(CONTINUOUS_COLUMNS)
  for i in range(0,continuous_len):
      continuous_cols.append(tf.contrib.layers.real_valued_column(CONTINUOUS_COLUMNS[i]))
  # Categorical base columns.
  categorical_cols = []
  categorical_len = len(CATEGORICAL_COLUMNS)
  for i in range(0,categorical_len):
      categorical_cols.append(tf.contrib.layers.real_valued_column(CONTINUOUS_COLUMNS[i]))


  # Wide columns and deep columns.
  wide_columns = categorical_columns
  deep_columns = continuous_columns

  if FLAGS.model_type == "wide":
    m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                          feature_columns=wide_columns)
  elif FLAGS.model_type == "deep":
    m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[0, 0, 0]) #this is something to consider
  else:
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])
    return m    
    
def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
    label = tf.constant(df['loss'].values)
    return continuous_cols, label
  
def train_input_fn():
    return input_fn(df_train)

def eval_input_fn():
    return input_fn(df_test)

def train_and_eval():
    model_dir = '/home/itzana/python/tensorflow/tf' if not FLAGS.model_dir else FLAGS.model_dir
    print("model directory = %s" % model_dir)
    m = build_estimator(model_dir)
    m.fit(input_fn=train_input_fn, steps=FLAGS.train_steps)
    results = m.evaluate(input_fn=eval_input_fn, steps=FLAGS.train_steps)
    for key in sorted(results):
        print "%s: %s" % (key, results[key])
        
def main(_):
  train_and_eval()


if __name__ == "__main__":
    tf.app.run()
