import os
from datetime import datetime

# model structure
from opt_classifier import *
# file io
from common.itbn_pipeline import *

TAG = "itbn_opt"

ALPHA = 1e-5
NUM_ITER = 30000

FRAME_SIZE = 45
STRIDE = 20
WINDOW_PROBABILITIES = [0.31, 0.12]
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


def chunk_data(frame_size, stride, seq_len, timing_dict, opt_raw):
    num_chunks = (seq_len - frame_size) / stride + 1
    opt_raw_out = None
    opt_label_data = None

    for i in range(num_chunks):
        opt_label_instance = label_data(frame_size, stride, i, seq_len, timing_dict)
        opt_raw_entry = np.expand_dims(opt_raw[0][stride * i:stride * i + frame_size], 0)

        if i == 0:
            opt_label_data = opt_label_instance
            opt_raw_out = opt_raw_entry
        else:
            opt_label_data = np.concatenate((opt_label_data, opt_label_instance), 0)
            opt_raw_out = np.concatenate((opt_raw_out, opt_raw_entry), 0)

    return opt_raw_out, opt_label_data


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
    dqn_chkpnt = ""
    dqn = ClassifierModel(batch_size=BATCH_SIZE, learning_rate=ALPHA, filename=dqn_chkpnt)

    # Train Model
    coord = tf.train.Coordinator()
    '''
    sequence length
    optical flow
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

    print("Num epochs: {}\nBatch Size: {}\nNum Files: {}\nNum iterations: {}".format(
        NUM_EPOCHS, BATCH_SIZE, len(filenames), NUM_ITER))

    balance_check = [0, 0, 0]
    last_dt = datetime.now()
    vals = {}
    opt_label_data = []
    use_window = False
    for iteration in range(NUM_ITER):
        # read a batch of tfrecords into np arrays
        seq_len, opt_raw, aud_raw, timing_labels, timing_values, name = dqn.sess.run(
            [seq_len_inp, opt_raw_inp, aud_raw_inp, timing_labels_inp, timing_values_inp, name_inp])
        timing_dict = parse_timing_dict(timing_labels[0], timing_values[0])
        num_chunks = (seq_len - FRAME_SIZE) / STRIDE + 1

        for i in range(num_chunks):
            # Label Data
            opt_label_data = label_data(FRAME_SIZE, STRIDE, i, seq_len, timing_dict)
            use_window = False
            window_class = int(np.argmax(opt_label_data))
            rand_val = np.random.random_sample()
            if window_class == 2 or rand_val < WINDOW_PROBABILITIES[window_class]:
                use_window = True
            if use_window:
                balance_check[window_class] += 1
                # Optimize Network
                vals = {
                    dqn.seq_length_ph: seq_len,
                    dqn.pnt_ph: np.expand_dims(opt_raw[0][STRIDE * i:STRIDE * i + FRAME_SIZE], 0),
                    dqn.pnt_y_ph: opt_label_data
                }

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                _ = dqn.sess.run([dqn.optimizer_wave],
                                 feed_dict=vals,
                                 options=run_options,
                                 run_metadata=run_metadata)

        # Print Metrics
        if iteration % 100 == 0:
            past_dt = last_dt
            last_dt = datetime.now()
            print("iteration: {}\ttime: {}\tclass counts: {}".format(
                iteration, last_dt - past_dt, balance_check))

        # evaluate system accuracy on train data set
        if iteration % 500 == 0:
            wave_pred = dqn.sess.run([dqn.wave_observed], feed_dict=vals)
            print("pred: {}\tlabel: {}".format(wave_pred[0][0], np.argmax(opt_label_data)))

        # Delayed System Updates
        if iteration % 5000 == 0:
            # save the model to checkpoint file
            dir_name = TAG + "_" + str(iteration / 5000)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            dqn.save_model(save_dir=dir_name)

    # FINISH
    # save final model to checkpoint file
    dir_name = TAG + "_final"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    dqn.save_model(save_dir=dir_name)

    print("time end: {}".format(datetime.now()))
