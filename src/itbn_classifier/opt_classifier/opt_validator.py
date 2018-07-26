# helper methods
import os
import sys
from datetime import datetime

# model structure
from opt_classifier import *
# file io
from common.itbn_pipeline import *

TAG = "itbn_opt"

ALPHA = 1e-5

FRAME_SIZE = 20
STRIDE = 7

SEQUENCE_CHARS = ["_", "|", "*"]
interval_relation_map = {
    (1., -1., -1., 1.): 'DURING',
    (-1., 1., -1., 1.): 'DURING_INV',
    (-1., -1., -1., 1.): 'OVERLAPS',
    (1., 1., -1., 1.): 'OVERLAPS_INV',
    (0., -1., -1., 1.): 'STARTS',
    (0., 1., -1., 1.): 'STARTS_INV',
    (1., 0., -1., 1.): 'FINISHES',
    (-1., 0., -1., 1.): 'FINISHES_INV',
    (0., 0., -1., 1.): 'EQUAL'
}


def calculate_relationship(window_s, window_e, event_s, event_e):
    temp_distance = (np.sign(event_s - window_s), np.sign(event_e - window_e),
                     np.sign(event_s - window_e), np.sign(event_e - window_s))
    return interval_relation_map.get(temp_distance, '')


def overlaps(s_time, e_time, td, label):
    s_label = label + "_s"
    e_label = label + "_e"

    # if s_label in td and ((td[s_label] <= s_time <= td[e_label]) or
    # (td[s_label] <= e_time <= td[e_label])):
    if s_label in td and calculate_relationship(s_time, e_time, td[s_label], td[e_label]) != '':
        return True
    return False


def label_data(frame_size, stride, frame_num, seq_len, timing_dict):
    opt_label_data = np.zeros((BATCH_SIZE, OPT_CLASSES)).astype(float)
    opt_label = 0

    s_frame = stride * frame_num
    e_frame = s_frame + frame_size

    if e_frame > seq_len:
        e_frame = seq_len

    if overlaps(s_frame, e_frame, timing_dict, "command"):
        opt_label = 1
    if overlaps(s_frame, e_frame, timing_dict, "prompt"):
        opt_label = 1
    if overlaps(s_frame, e_frame, timing_dict, "noise_0"):
        opt_label = 1
    if overlaps(s_frame, e_frame, timing_dict, "noise_1"):
        opt_label = 1
    if overlaps(s_frame, e_frame, timing_dict, "gesture_0"):
        opt_label = 2
    if overlaps(s_frame, e_frame, timing_dict, "gesture_1"):
        opt_label = 2

    opt_label_data[0][opt_label] = 1
    return opt_label_data


if __name__ == '__main__':
    print("time start: {}".format(datetime.now()))

    # Read contents of TFRecord file
    # generate list of filenames
    path = "../../ITBN_tfrecords/"
    filenames = list()
    for root, dir, files in os.walk(path):
        for f in files:
            if 'validation' not in f:
                filenames.append(os.path.join(root, f))
    filenames.sort()

    print("{}".format(filenames))

    # Generate Model
    # if building model from a checkpoint define location here. Otherwise use empty string ""
    dqn_chkpnt = "itbn_opt_final/model.ckpt"
    dqn = ClassifierModel(batch_size=BATCH_SIZE, learning_rate=ALPHA, filename=dqn_chkpnt)

    # Train Model
    coord = tf.train.Coordinator()
    '''
    sequence length
    optical raw
    audio raw
    timing values
    file identifier
    '''
    # read records from files into tensors
    seq_len_inp, opt_raw_inp, aud_raw_inp, timing_labels_inp, timing_values_inp, name_inp = \
        input_pipeline(filenames)

    # initialize all variables
    dqn.sess.run(tf.local_variables_initializer())
    dqn.sess.graph.finalize()
    threads = tf.train.start_queue_runners(coord=coord, sess=dqn.sess)

    print("Num epochs: {}\nBatch Size: {}\nNum Files: {}".format(
        NUM_EPOCHS, BATCH_SIZE, len(filenames)))

    matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    num_files = len(filenames)
    counter = 0
    sequences = dict()
    while len(filenames) > 0:
        # read a batch of tfrecords into np arrays
        seq_len, opt_raw, aud_raw, timing_labels, timing_values, name = dqn.sess.run(
            [seq_len_inp, opt_raw_inp, aud_raw_inp, timing_labels_inp, timing_values_inp, name_inp])
        name = name[0].replace('.txt', '.tfrecord').replace(
            '/home/assistive-robotics/PycharmProjects/dbn_arl/labels/', '../../ITBN_tfrecords/')
        if name in filenames:
            counter += 1
            print("processing {}/{}: {}".format(counter, num_files, name))
            filenames.remove(name)
            timing_dict = parse_timing_dict(timing_labels[0], timing_values[0])
            num_chunks = (seq_len - FRAME_SIZE) / STRIDE + 1
            real_sequence = ""
            pred_sequence = ""

            for i in range(num_chunks):
                # Label Data
                opt_label_data = label_data(FRAME_SIZE, STRIDE, i, seq_len, timing_dict)
                vals = {
                    dqn.seq_length_ph: seq_len,
                    dqn.pnt_ph: np.expand_dims(opt_raw[0][STRIDE * i:STRIDE * i + FRAME_SIZE], 0),
                    dqn.pnt_y_ph: opt_label_data
                }
                opt_pred = dqn.sess.run([dqn.wave_observed], feed_dict=vals)
                real_class = int(np.argmax(opt_label_data))
                selected_class = int(opt_pred[0][0])
                matrix[real_class][selected_class] += 1
                real_sequence += SEQUENCE_CHARS[real_class]
                pred_sequence += SEQUENCE_CHARS[selected_class]
            sequences[name] = real_sequence + "\n" + pred_sequence

    print("time end: {}\n{}\n".format(datetime.now(), matrix))

    strings = ['.tfrecord', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_', '-', "../../ITBN_tfrecords/test_", 'validation']
    for f in sequences.keys():
        original = f
        # for s in strings:
        #     f = f.replace(s, '')
        # if 'a' in f:
        print("{}\n{}\n".format(original, sequences[original]))
