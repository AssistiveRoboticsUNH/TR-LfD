# model structure
import os

from common.itbn_classifier import *
# file io
from common.itbn_pipeline import *

# helper methods

ALPHA = 1e-5
FRAME_SIZE = 45
STRIDE = 20
FRAME_SIZE_AUD = 20
STRIDE_AUD = 7


#OPT 45,20  -   probabilities: 0.31, 0.12, 1.0
#AUD 20, 7  -   probabilities: 0.08, 0.03, 1.0

def overlaps(s_time, e_time, td, label):
    s_label = label + "_s"
    e_label = label + "_e"

    if (s_label in td and
            ((td[s_label] <= s_time <= td[e_label])
             or (td[s_label] <= e_time <= td[e_label]))):
        return True
    return False


def label_data(FRAME_SIZE, STRIDE, i, seq_len, timing_dict):
    #OPT
    #0 = nothing
    #1 = command, prompt, noise
    #2 = gesture
    #AUD
    #0 = nothing
    #1 = command, prompt, reward, abort, noise
    #2 = audio
    opt_label_data = np.zeros((BATCH_SIZE, OPT_CLASSES)).astype(float)
    aud_label_data = np.zeros((BATCH_SIZE, AUD_CLASSES)).astype(float)
    opt_label = 0
    aud_label = 0

    s_frame = STRIDE * i
    e_frame = s_frame + FRAME_SIZE

    if (e_frame > seq_len):
        e_frame = seq_len

    if (overlaps(s_frame, e_frame, timing_dict, "noise_0")):
        opt_label, aud_label = 1, 1
    if (overlaps(s_frame, e_frame, timing_dict, "noise_1")):
        opt_label, aud_label = 1, 1

    if (overlaps(s_frame, e_frame, timing_dict, "command")):
        opt_label, aud_label = 1, 1
    if (overlaps(s_frame, e_frame, timing_dict, "prompt")):
        opt_label, aud_label = 1, 1
    if (overlaps(s_frame, e_frame, timing_dict, "reward")):
        aud_label = 1
    if (overlaps(s_frame, e_frame, timing_dict, "abort")):
        aud_label = 1

    if (overlaps(s_frame, e_frame, timing_dict, "gesture_0")):
        opt_label = 2
    if (overlaps(s_frame, e_frame, timing_dict, "gesture_1")):
        opt_label = 2

    if (overlaps(s_frame, e_frame, timing_dict, "audio_0")):
        aud_label = 2
    if (overlaps(s_frame, e_frame, timing_dict, "audio_1")):
        aud_label = 2

    opt_label_data[0][opt_label] = 1
    aud_label_data[0][aud_label] = 1
    return opt_label_data, aud_label_data


def chunk_data(FRAME_SIZE, STRIDE, seq_len, timing_dict, opt_raw, aud_raw):
    num_chunks = (seq_len - FRAME_SIZE) / STRIDE + 1

    opt_raw_out, aud_raw_out = None, None
    opt_label_data, aud_label_data = None, None

    for i in range(num_chunks):
        opt_label_instance, aud_label_instance = label_data(FRAME_SIZE, STRIDE, i, seq_len,
                                                            timing_dict)

        opt_raw_entry = np.expand_dims(opt_raw[0][STRIDE * i:STRIDE * i + FRAME_SIZE], 0)
        aud_raw_entry = np.expand_dims(aud_raw[0][STRIDE * i:STRIDE * i + FRAME_SIZE], 0)

        if (i == 0):
            opt_label_data = opt_label_instance
            aud_label_data = aud_label_instance

            opt_raw_out = opt_raw_entry
            aud_raw_out = aud_raw_entry
        else:

            opt_label_data = np.concatenate((opt_label_data, opt_label_instance), 0)
            aud_label_data = np.concatenate((aud_label_data, aud_label_instance), 0)

            opt_raw_out = np.concatenate((opt_raw_out, opt_raw_entry), 0)
            aud_raw_out = np.concatenate((aud_raw_out, aud_raw_entry), 0)

    return opt_raw_out, aud_raw_out, opt_label_data, aud_label_data


if __name__ == '__main__':
    graphbuild = [1] * TOTAL_PARAMS
    num_params = np.sum(graphbuild)

    # Read contents of TFRecord file
    path = "../ITBN_tfrecords/"

    # generate list of filenames
    filenames = list()
    for root, dir, files in os.walk(path):
        for f in files:
            if 'validation' not in f:
                filenames.append(os.path.join(root, f))
    filenames.sort()

    # Generate Model
    dqn_chkpnt = ""
    dqn = ClassifierModel(batch_size=BATCH_SIZE, win_size=FRAME_SIZE, learning_rate=ALPHA,
                          filename=dqn_chkpnt, log_dir="LOG_DIR")

    # Train Model
    coord = tf.train.Coordinator()
    '''
    sequence length - slen
    sequence length prime- slen_pr
    image raw - i
    points raw - p
    audio raw - a
    previous action - pl
    action - l
    image raw prime - i_pr
    points raw prime - p_pr
    audio raw prime - a_pr
    file identifier - n_id
    '''
    # read records from files into tensors
    seq_len_inp, opt_raw_inp, aud_raw_inp, timing_labels_inp, timing_values_inp, name_inp = \
        input_pipeline(filenames)

    # initialize all variables
    dqn.sess.run(tf.local_variables_initializer())
    dqn.sess.graph.finalize()
    threads = tf.train.start_queue_runners(coord=coord, sess=dqn.sess)

    counter = 0
    avg_chunks = 0
    empty_count_opt = 0
    empty_count_aud = 0
    robot_motion = 0
    human_gesture = 0
    robot_audio = 0
    human_audio = 0
    while len(filenames) > 0:
        # read a batch of tfrecords into np arrays
        seq_len, opt_raw, aud_raw, timing_labels, timing_values, name = dqn.sess.run(
            [seq_len_inp, opt_raw_inp, aud_raw_inp, timing_labels_inp, timing_values_inp, name_inp])
        name = name[0].replace('.txt', '.tfrecord').replace(
            '/home/assistive-robotics/PycharmProjects/dbn_arl/labels/', '../ITBN_tfrecords/')
        if name in filenames:
            # print('{}'.format(name))
            filenames.remove(name)
            timing_dict = parse_timing_dict(timing_labels[0], timing_values[0])
            num_chunks = (seq_len - FRAME_SIZE) / STRIDE + 1
            for i in range(num_chunks):
                # Label Data
                # one hot representation of the state or action
                opt_label_data, aud_label_data = label_data(FRAME_SIZE, STRIDE, i,
                                                            seq_len, timing_dict)
                # print(opt_label_data, np.argmax(opt_label_data))
                if np.argmax(opt_label_data) == 0:
                    empty_count_opt += 1
                if np.argmax(opt_label_data) == 1:
                    robot_motion += 1
                if np.argmax(opt_label_data) == 2:
                    human_gesture += 1
            avg_chunks += num_chunks
            counter += 1
    print('number of files: {}\nnumber of chunks: {}\nempty: {}\nrobot: {}\ngesture: {}'.format(
        counter, avg_chunks, empty_count_opt, robot_motion, human_gesture))

    filenames = list()
    for root, dir, files in os.walk(path):
        for f in files:
            if 'validation' not in f:
                filenames.append(os.path.join(root, f))
    filenames.sort()
    while len(filenames) > 0:
        # read a batch of tfrecords into np arrays
        seq_len, opt_raw, aud_raw, timing_labels, timing_values, name = dqn.sess.run(
            [seq_len_inp, opt_raw_inp, aud_raw_inp, timing_labels_inp, timing_values_inp, name_inp])
        name = name[0].replace('.txt', '.tfrecord').replace(
            '/home/assistive-robotics/PycharmProjects/dbn_arl/labels/', '../ITBN_tfrecords/')
        if name in filenames:
            # print('{}'.format(name))
            filenames.remove(name)
            timing_dict = parse_timing_dict(timing_labels[0], timing_values[0])
            num_chunks = (seq_len - FRAME_SIZE_AUD) / STRIDE_AUD + 1
            for i in range(num_chunks):
                # Label Data
                # one hot representation of the state or action
                opt_label_data, aud_label_data = label_data(FRAME_SIZE_AUD, STRIDE_AUD, i,
                                                            seq_len, timing_dict)
                # print(opt_label_data, np.argmax(opt_label_data))
                if np.argmax(aud_label_data) == 0:
                    empty_count_aud += 1
                if np.argmax(aud_label_data) == 1:
                    robot_audio += 1
                if np.argmax(aud_label_data) == 2:
                    human_audio += 1
    print('\nempty: {}\nrobot: {}\naudio: {}'.format(empty_count_aud, robot_audio, human_audio))
