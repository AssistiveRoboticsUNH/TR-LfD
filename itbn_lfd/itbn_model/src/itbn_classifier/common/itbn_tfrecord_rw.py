#!/usr/bin/env python

# itbn_tfrecord_rw.py
# Madison Clark-Turner
# 12/2/2017

import numpy as np
import tensorflow as tf

from itbn_classifier.common.constants import *


# WRITE
def make_sequence_example(img_input, img_data, opt_input, opt_data,
                          aud_input, aud_data, timing_dict, example_id):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = opt_input.shape[1]

    ex.context.feature["length"].int64_list.value.append(sequence_length)

    # ex.context.feature["img_h"].int64_list.value.append(img_data["cmp_h"])
    # ex.context.feature["img_w"].int64_list.value.append(img_data["cmp_w"])
    # ex.context.feature["img_c"].int64_list.value.append(img_data["num_c"])

    ex.context.feature["pnt_h"].int64_list.value.append(opt_data["cmp_h"])
    ex.context.feature["pnt_w"].int64_list.value.append(opt_data["cmp_w"])
    ex.context.feature["pnt_c"].int64_list.value.append(opt_data["num_c"])

    ex.context.feature["aud_h"].int64_list.value.append(aud_data["cmp_h"])
    ex.context.feature["aud_w"].int64_list.value.append(aud_data["cmp_w"])
    ex.context.feature["aud_c"].int64_list.value.append(aud_data["num_c"])

    ex.context.feature["example_id"].bytes_list.value.append(example_id)

    timing_labels, timing_values = "", []
    for k in timing_dict.keys():
        timing_labels += k + "/"
        timing_values.append(timing_dict[k])

    ex.context.feature["timing_labels"].bytes_list.value.append(timing_labels)

    # Feature lists for input data
    def load_array(example, name, data, dtype):
        fl_data = example.feature_lists.feature_list[name].feature.add().bytes_list.value
        fl_data.append(np.asarray(data).astype(dtype).tostring())

    # load_array(ex, "img_raw", img_input, np.uint8)
    load_array(ex, "opt_raw", opt_input, np.uint8)
    load_array(ex, "aud_raw", aud_input, np.uint8)
    load_array(ex, "timing_values", timing_values, np.int16)

    return ex


# READ
def parse_sequence_example(filename_queue):
    # reads a TFRecord into its constituent parts
    reader = tf.TFRecordReader()
    _, example = reader.read(filename_queue)

    context_features = {
        "length": tf.FixedLenFeature([], dtype=tf.int64),

        # "img_h": tf.FixedLenFeature([], dtype=tf.int64),
        # "img_w": tf.FixedLenFeature([], dtype=tf.int64),
        # "img_c": tf.FixedLenFeature([], dtype=tf.int64),

        # "pnt_h": tf.FixedLenFeature([], dtype=tf.int64),
        # "pnt_w": tf.FixedLenFeature([], dtype=tf.int64),
        # "pnt_c": tf.FixedLenFeature([], dtype=tf.int64),

        # "aud_h": tf.FixedLenFeature([], dtype=tf.int64),
        # "aud_w": tf.FixedLenFeature([], dtype=tf.int64),
        # "aud_c": tf.FixedLenFeature([], dtype=tf.int64),

        "example_id": tf.FixedLenFeature([], dtype=tf.string),

        "timing_labels": tf.FixedLenFeature([], dtype=tf.string),
    }

    sequence_features = {
        # "img_raw": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "opt_raw": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "aud_raw": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "timing_values": tf.FixedLenSequenceFeature([], dtype=tf.string)
    }

    # Parse the example
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    context_data = context_parsed
    context_data["timing_labels"] = tf.string_split(
        tf.expand_dims(context_parsed["timing_labels"], 0), delimiter='/').values

    sequence_data = {
        # "img_raw": tf.decode_raw(sequence_parsed["img_raw"], tf.uint8),
        "opt_raw": tf.decode_raw(sequence_parsed["opt_raw"], tf.uint8),
        "aud_raw": tf.decode_raw(sequence_parsed["aud_raw"], tf.uint8),
        "timing_values": tf.decode_raw(sequence_parsed["timing_values"], tf.int16)  # timing_values
    }

    return context_data, sequence_data


def set_input_shape(arr, data_type):
    return np.reshape(arr,
                      (BATCH_SIZE, -1, data_type["size"] * data_type["size"] * data_type["num_c"]))


def parse_timing_dict(labels, values):
    return dict(zip(labels, values[0]))
