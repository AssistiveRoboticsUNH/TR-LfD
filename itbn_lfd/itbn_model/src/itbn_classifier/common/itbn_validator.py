# model_trainer_omega.py
# Madison Clark-Turner
# 10/13/2017

TAG = "itbn"

import os
# helper methods
import sys
from datetime import datetime

# model structure
from common.itbn_classifier import *
# file io
from common.itbn_pipeline import *

GAMMA = 0.9
ALPHA = 1e-5
NUM_ITER = 30000
FOLDS = 1
NUM_REMOVED = 1

TEST_ITER = 50

FRAME_SIZE = 50
STRIDE = 25


def overlaps(s_time, e_time, td, label):
    s_label = label + "_s"
    e_label = label + "_e"

    if (s_label in td and
            ((s_time >= td[s_label] and s_time <= td[e_label])
             or (e_time >= td[s_label] and e_time <= td[e_label]))):
        return True
    return False


def label_data(FRAME_SIZE, STRIDE, i, seq_len, timing_dict):
    opt_label_data = np.zeros((BATCH_SIZE, OPT_CLASSES)).astype(float)
    aud_label_data = np.zeros((BATCH_SIZE, AUD_CLASSES)).astype(float)
    opt_label = 0
    aud_label = 0

    s_frame = STRIDE * i
    e_frame = s_frame + FRAME_SIZE

    if (e_frame > seq_len):
        e_frame = seq_len

    if (overlaps(s_frame, e_frame, timing_dict, "command")):
        opt_label, aud_label = 1, 1
    if (overlaps(s_frame, e_frame, timing_dict, "prompt")):
        opt_label, aud_label = 1, 1
    if (overlaps(s_frame, e_frame, timing_dict, "reward")):
        opt_label, aud_label = 1, 1
    if (overlaps(s_frame, e_frame, timing_dict, "abort")):
        opt_label, aud_label = 1, 1

    if (overlaps(s_frame, e_frame, timing_dict, "noise_0")):
        aud_label = 1
    if (overlaps(s_frame, e_frame, timing_dict, "noise_1")):
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

    ts = datetime.now()
    print("time start: ", str(ts))
    #################################
    # Command-Line Parameters
    #################################
    graphbuild = [0] * TOTAL_PARAMS
    if (len(sys.argv) > 1):
        if (len(sys.argv[1]) == 1 and int(sys.argv[1]) < 3):
            graphbuild[int(sys.argv[1])] = 1
        else:
            print("Usage: python model_trainer_omega.py <args>")
            print("\t0 - only build network with RGB information")
            print("\t1 - only build network with Optical Flow information")
            print("\t2 - only build network with Audio information")
            print("\t(nothing) - build network with all information")
    else:
        graphbuild = [1] * TOTAL_PARAMS

    if (sum(graphbuild) < 3):
        print("#########################")
        print("BUILDING PARTIAL MODEL")
        print("#########################")

    num_params = np.sum(graphbuild)

    #################################
    # Read contents of TFRecord file
    #################################

    # define directory to read files from
    path = "../ITBN_tfrecords/"

    filenames = list()
    # generate list of filenames
    for root, dir, files in os.walk(path):
        for f in files:
            if 'validation' in f:
                filenames.append(os.path.join(root, f))
    filenames.sort()

    print("Filenames: ", filenames)

    #################################
    # Generate Model
    #################################

    # if building model from a checkpoint define location here. Otherwise use empty string ""
    dqn_chkpnt = "../ITBN_past_models/itbn_1"
    dqn = ClassifierModel(batch_size=BATCH_SIZE, win_size=FRAME_SIZE, learning_rate=ALPHA,
                          filename=dqn_chkpnt, \
                          log_dir="LOG_DIR")

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
    seq_len_inp, opt_raw_inp, aud_raw_inp, timing_labels_inp, timing_values_inp, name_inp = input_pipeline(
        filenames)

    # initialize all variables
    dqn.sess.run(tf.local_variables_initializer())
    dqn.sess.graph.finalize()
    threads = tf.train.start_queue_runners(coord=coord, sess=dqn.sess)

    print("Num epochs: " + str(NUM_EPOCHS) + ", Batch Size: " + str(BATCH_SIZE) + ", Num Files: " + \
          str(len(filenames)) + ", Num iterations: " + str(NUM_ITER))

    for iteration in range(NUM_ITER):
        ts_it = datetime.now()

        # ---------------------------------------
        # read a bacth of tfrecords into np arrays
        # ---------------------------------------

        seq_len, opt_raw, aud_raw, timing_labels, timing_values, name = dqn.sess.run(
            [seq_len_inp, opt_raw_inp, aud_raw_inp, timing_labels_inp, timing_values_inp, name_inp])

        timing_dict = parse_timing_dict(timing_labels[0], timing_values[0])

        # print(timing_dict.keys())

        # ---------------------------------------
        # generate partitions; used for extracting relevant data from the LSTM layer
        # ---------------------------------------

        prep_t = datetime.now() - ts_it

        num_chunks = (seq_len - FRAME_SIZE) / STRIDE + 1

        opt_t = datetime.now() - datetime.now()

        # print()
        for i in range(num_chunks):
            # if (i % 100 == 0):
            #     print(
            #     "Processing chunk: " + str(i) + "/" + str(num_chunks[0]) + " of file: " + name[0])
            # ---------------------------------------
            # Label Data
            # ---------------------------------------

            opt_label_data, aud_label_data = label_data(FRAME_SIZE, STRIDE, i, seq_len,
                                                        timing_dict)  # one hot representation of the state or action
            # print(opt_label_data, aud_label_data)
            # ---------------------------------------
            # Optimize Network
            # ---------------------------------------

            vals = {
                dqn.seq_length_ph: seq_len,
                dqn.pnt_ph: np.expand_dims(opt_raw[0][STRIDE * i:STRIDE * i + FRAME_SIZE], 0),
                dqn.aud_ph: np.expand_dims(aud_raw[0][STRIDE * i:STRIDE * i + FRAME_SIZE], 0),
                dqn.pnt_y_ph: opt_label_data,
                dqn.aud_y_ph: aud_label_data,
            }

            # OPTIMIZE
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            opt_s = datetime.now()

            # optimize network
            _ = dqn.sess.run([dqn.optimizer_wave, dqn.optimizer_aud], feed_dict=vals,
                             options=run_options,
                             run_metadata=run_metadata)

            opt_t += datetime.now() - opt_s

        '''        
            vals = {
                dqn.seq_length_ph: seq_len, 
                dqn.pnt_ph: opt_raw[chunk], 
                dqn.aud_ph: aud_raw[chunk], 
                dqn.pnt_y_ph: opt_label_data[chunk],
                dqn.aud_y_ph: aud_label_data[chunk],
                        
            # OPTIMIZE
            run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    
            opt_s = datetime.now()
            
            # optimize network
            _ = dqn.sess.run([dqn.optimizer_wave, dqn.optimizer_aud], feed_dict=vals, options=run_options, run_metadata=run_metadata)
            
            opt_t += datetime.now() - opt_s
        '''
        # ---------------------------------------
        # Print Metrics
        # ---------------------------------------
        if (iteration % 10 == 0):
            print(datetime.now())
            # print timing information
            print(iteration, "total_time:", str(datetime.now() - ts_it), "prep_time:", str(prep_t),
                  "optimization_time:", str(opt_t))

        if (iteration % 100 == 0):
            # evaluate system accuracy on train dataset
            wave_pred, aud_pred = dqn.sess.run([dqn.wave_observed, dqn.aud_observed],
                                               feed_dict=vals)
            '''
            print("pred: ", pred)
            print("label: ", label_data)
            print("--------")
            
            acc = dqn.sess.run(dqn.accuracy, feed_dict=vals)
            print("acc of train: ", acc)
            '''
        # ---------------------------------------
        # Delayed System Updates
        # ---------------------------------------

        if (iteration % 5000 == 0):
            # save the model to checkpoint file
            # overwrite the saved model until 10,000 iterations have passed
            dir_name = TAG + "_" + str(iteration / 5000)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            dqn.saveModel(save_dir=dir_name)

    #######################
    ## FINISH
    #######################

    # save final model to chekpoint file
    dir_name = TAG + "_final"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    dqn.saveModel(save_dir=dir_name)

    te = datetime.now()
    print("time end: ", te)
    print("elapsed: ", te - ts)
