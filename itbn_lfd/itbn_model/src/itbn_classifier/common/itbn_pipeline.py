from common.constants import *
from itbn_tfrecord_rw import *


def input_pipeline(filenames):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=NUM_EPOCHS, shuffle=True)

    min_after_dequeue = 7  # buffer to shuffle with (bigger=better shuffeling) #7
    capacity = min_after_dequeue + 3  # * BATCH_SIZE

    # deserialize is a custom function that deserializes string
    # representation of a single tf.train.Example into tensors
    # (features, label) representing single training example
    context_parsed, sequence_parsed = parse_sequence_example(filename_queue)

    seq_len = context_parsed["length"]
    timing_labels = context_parsed["timing_labels"]
    timing_values = sequence_parsed["timing_values"]
    name = context_parsed["example_id"]

    def processData(inp, data_type):
        data_s = tf.reshape(inp, [-1, data_type["cmp_h"] * data_type["cmp_w"] * data_type["num_c"]])
        return tf.cast(data_s, tf.uint8)

    # img_raw = processData(sequence_parsed["img_raw"], img_dtype)
    opt_raw = processData(sequence_parsed["opt_raw"], pnt_dtype)
    aud_raw = processData(sequence_parsed["aud_raw"], aud_dtype)

    # Imagine inputs is a list or tuple of tensors representing single training example.
    # In my case, inputs is a tuple (features, label) obtained by reading TFRecords.
    NUM_THREADS = 1
    QUEUE_RUNNERS = 1

    inputs = [seq_len, opt_raw, aud_raw, timing_labels, timing_values, name]

    dtypes = list(map(lambda x: x.dtype, inputs))
    shapes = list(map(lambda x: x.get_shape(), inputs))

    queue = tf.RandomShuffleQueue(capacity, min_after_dequeue, dtypes)

    enqueue_op = queue.enqueue(inputs)
    qr = tf.train.QueueRunner(queue, [enqueue_op] * NUM_THREADS)

    tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, qr)
    inputs = queue.dequeue()

    for tensor, shape in zip(inputs, shapes):
        tensor.set_shape(shape)

    inputs_batch = tf.train.batch(inputs,
                                  BATCH_SIZE,
                                  capacity=capacity,
                                  dynamic_pad=True
                                  )

    return inputs_batch[0], inputs_batch[1], inputs_batch[2], inputs_batch[3], inputs_batch[4], \
           inputs_batch[5]
    # seq_len,        #points,         #audio_raw,      #timing_labels,  #timing_values,  #example_id


def set_shape(arr, data_type):
    return np.reshape(arr, (
    BATCH_SIZE, -1, data_type["cmp_h"] * data_type["cmp_w"] * data_type["num_c"]))
