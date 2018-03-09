# helper methods
import os
from datetime import datetime

# model structure
from opt_classifier import opt_classifier
from aud_classifier import aud_classifier

# file io
from common.itbn_pipeline import *

AUD_TAG = "itbn_aud"
OPT_TAG = "itbn_opt"

ALPHA = 1e-5

AUD_FRAME_SIZE = 20
AUD_STRIDE = 7
OPT_FRAME_SIZE = 45
OPT_STRIDE = 20

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

validation = False


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


def label_data_aud(frame_size, stride, frame_num, seq_len, timing_dict):
    aud_label_data = np.zeros((BATCH_SIZE, AUD_CLASSES)).astype(float)
    aud_label = 0

    s_frame = stride * frame_num
    e_frame = s_frame + frame_size

    if e_frame > seq_len:
        e_frame = seq_len

    if overlaps(s_frame, e_frame, timing_dict, "command"):
        aud_label = 1
    if overlaps(s_frame, e_frame, timing_dict, "prompt"):
        aud_label = 1
    if overlaps(s_frame, e_frame, timing_dict, "reward"):
        aud_label = 1
    if overlaps(s_frame, e_frame, timing_dict, "abort"):
        aud_label = 1
    if overlaps(s_frame, e_frame, timing_dict, "noise_0"):
        aud_label = 1
    if overlaps(s_frame, e_frame, timing_dict, "noise_1"):
        aud_label = 1
    if overlaps(s_frame, e_frame, timing_dict, "audio_0"):
        aud_label = 2
    if overlaps(s_frame, e_frame, timing_dict, "audio_1"):
        aud_label = 2

    aud_label_data[0][aud_label] = 1
    return aud_label_data


def label_data_opt(frame_size, stride, frame_num, seq_len, timing_dict):
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
    path = "../ITBN_tfrecords/"
    filenames = list()
    for root, dir, files in os.walk(path):
        for f in files:
            if validation:
                if 'validation' in f:
                    filenames.append(os.path.join(root, f))
            else:
                if 'validation' not in f:
                    filenames.append(os.path.join(root, f))
    filenames.sort()

    print("{}".format(filenames))

    # Generate Model
    # if building model from a checkpoint define location here. Otherwise use empty string ""
    aud_dqn_chkpnt = "aud_classifier/itbn_aud_final/model.ckpt"
    opt_dqn_chkpnt = "opt_classifier/itbn_opt_final/model.ckpt"
    aud_dqn = aud_classifier.ClassifierModel(batch_size=BATCH_SIZE, learning_rate=ALPHA, filename=aud_dqn_chkpnt)
    opt_dqn = opt_classifier.ClassifierModel(batch_size=BATCH_SIZE, learning_rate=ALPHA, filename=opt_dqn_chkpnt)

    # Train Model
    aud_coord = tf.train.Coordinator()
    opt_coord = tf.train.Coordinator()
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
    with aud_dqn.sess.as_default():
        with aud_dqn.graph.as_default():
            aud_dqn.sess.run(tf.local_variables_initializer())
            aud_dqn.sess.graph.finalize()
            threads = tf.train.start_queue_runners(coord=aud_coord, sess=aud_dqn.sess)

    with opt_dqn.sess.as_default():
        with opt_dqn.graph.as_default():
            opt_dqn.sess.run(tf.local_variables_initializer())
            opt_dqn.sess.graph.finalize()
            threads = tf.train.start_queue_runners(coord=opt_coord, sess=opt_dqn.sess)

    print("Num epochs: {}\nBatch Size: {}\nNum Files: {}".format(NUM_EPOCHS, BATCH_SIZE, len(filenames)))

    aud_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    opt_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    num_files = len(filenames)
    counter = 0
    aud_sequences = dict()
    opt_sequences = dict()
    while len(filenames) > 0:
        # read a batch of tfrecords into np arrays
        seq_len, opt_raw, aud_raw, timing_labels, timing_values, name = opt_dqn.sess.run(
            [seq_len_inp, opt_raw_inp, aud_raw_inp, timing_labels_inp, timing_values_inp, name_inp])

        if validation:
            name = name[0].replace('.txt', '_validation.tfrecord').replace(
                '/home/assistive-robotics/PycharmProjects/dbn_arl/labels/', '../ITBN_tfrecords/')
        else:
            name = name[0].replace('.txt', '.tfrecord').replace(
                '/home/assistive-robotics/PycharmProjects/dbn_arl/labels/', '../ITBN_tfrecords/')
        if name in filenames:
            counter += 1
            print("processing {}/{}: {}".format(counter, num_files, name))
            filenames.remove(name)
            timing_dict = parse_timing_dict(timing_labels[0], timing_values[0])
            aud_num_chunks = (seq_len - AUD_FRAME_SIZE) / AUD_STRIDE + 1
            opt_num_chunks = (seq_len - OPT_FRAME_SIZE) / OPT_STRIDE + 1
            aud_real_sequence = ""
            aud_pred_sequence = ""
            opt_real_sequence = ""
            opt_pred_sequence = ""

            for i in range(max(opt_num_chunks, aud_num_chunks)):
                if i < aud_num_chunks:
                    with aud_dqn.sess.as_default():
                        # Label Data
                        aud_label_data = label_data_aud(AUD_FRAME_SIZE, AUD_STRIDE, i, seq_len, timing_dict)
                        vals = {
                            aud_dqn.seq_length_ph: seq_len,
                            aud_dqn.aud_ph: np.expand_dims(aud_raw[0][AUD_STRIDE * i:AUD_STRIDE * i + AUD_FRAME_SIZE], 0),
                            aud_dqn.aud_y_ph: aud_label_data
                        }
                        aud_pred = aud_dqn.sess.run([aud_dqn.aud_observed], feed_dict=vals)
                        real_class = int(np.argmax(aud_label_data))
                        selected_class = int(aud_pred[0][0])
                        aud_matrix[real_class][selected_class] += 1
                        aud_real_sequence += SEQUENCE_CHARS[real_class]
                        aud_pred_sequence += SEQUENCE_CHARS[selected_class]
                if i < opt_num_chunks:
                    with opt_dqn.sess.as_default():
                        # Label Data
                        opt_label_data = label_data_opt(OPT_FRAME_SIZE, OPT_STRIDE, i, seq_len, timing_dict)
                        vals = {
                            opt_dqn.seq_length_ph: seq_len,
                            opt_dqn.pnt_ph: np.expand_dims(opt_raw[0][OPT_STRIDE * i:OPT_STRIDE * i + OPT_FRAME_SIZE], 0),
                            opt_dqn.pnt_y_ph: opt_label_data
                        }
                        opt_pred = opt_dqn.sess.run([opt_dqn.wave_observed], feed_dict=vals)
                        real_class = int(np.argmax(opt_label_data))
                        selected_class = int(opt_pred[0][0])
                        opt_matrix[real_class][selected_class] += 1
                        opt_real_sequence += SEQUENCE_CHARS[real_class]
                        opt_pred_sequence += SEQUENCE_CHARS[selected_class]
            aud_sequences[name] = aud_real_sequence + "\n" + aud_pred_sequence
            opt_sequences[name] = opt_real_sequence + "\n" + opt_pred_sequence

    print("time end: {}\nAUDIO\n{}\n\nVIDEO\n{}\n".format(datetime.now(), aud_matrix, opt_matrix))

    strings = ['.tfrecord', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_', '-', "../../ITBN_tfrecords/test_", 'validation']
    print("\n\nAUDIO SEQUENCES:")
    for f in aud_sequences.keys():
        print("{}\n{}\n".format(f, aud_sequences[f]))

    print("\n\nVIDEO SEQUENCES:")
    for f in opt_sequences.keys():
        print("{}\n{}\n".format(f, opt_sequences[f]))
