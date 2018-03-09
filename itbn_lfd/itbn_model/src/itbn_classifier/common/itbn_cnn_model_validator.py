# IMPORTS ##############################################
import os
import networkx as nx
import pandas as pd
from datetime import datetime

# cnn models
from opt_classifier import opt_classifier
from aud_classifier import aud_classifier

# file io
from common.itbn_pipeline import *

# itbn model
from pgmpy.models import IntervalTemporalBayesianNetwork

# PARAMETERS ##############################################

# disable unwanted pandas warnings
pd.options.mode.chained_assignment = None

# boolean used to switch between training/validation
validation = True

# itbn model path
ITBN_MODEL_PATH = 'input/itbn.nx'

# cnn models paths
AUD_DQN_CHKPNT = "../aud_classifier/itbn_aud_final/model.ckpt"
OPT_DQN_CHKPNT = "../opt_classifier/itbn_opt_final/model.ckpt"

# tf records path
TF_RECORDS_PATH = "../../ITBN_tfrecords/"
LABELS_PATH = '/home/assistive-robotics/PycharmProjects/dbn_arl/labels/'

# cnn parameters
ALPHA = 1e-5
AUD_FRAME_SIZE = 20
AUD_STRIDE = 7
OPT_FRAME_SIZE = 20 #45
OPT_STRIDE = 7 #20
FAR_FRAME = 10000

# debug characters for cnn classifications [silence, robot, human]
SEQUENCE_CHARS = ["_", "|", "*"]

# subset of allen's interval relations that are used to classify a window fed to the cnns
WINDOW_INTERVAL_RELATION_MAP = {
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
EVENT_INTERVAL_RELATION_MAP = {
    (-1., -1., -1., -1.): 1,
    (1., 1., 1., 1.): 2,
    (-1., -1., -1., 0.): 1,
    (1., 1., 0., 1.): 2,
    (-1., -1., -1., 1.): 1,
    (1., 1., -1., 1.): 2,
    (1., -1., -1., 1.) : 7,
    (-1., 1., -1., 1.) : 8,
    (0., -1., -1., 1.) : 9,
    (0., 1., -1., 1.) : 10,
    (1., 0., -1., 1.) : 11,
    (-1., 0., -1., 1.) : 12,
    (0., 0., -1., 1.) : 13
}
EVALUATION_INTERVAL_RELATION_MAP = {
    (-1., -1., -1., 1.): 5,
    (1., 1., -1., 1.): 6,
    (1., -1., -1., 1.): 7,
    (-1., 1., -1., 1.): 8,
    (0., -1., -1., 1.): 9,
    (0., 1., -1., 1.): 10,
    (1., 0., -1., 1.): 11,
    (-1., 0., -1., 1.) : 12,
    (0., 0., -1., 1.) : 13
}


# FUNCTIONS DEFINITION ##############################################
# given a window and event start and end time calculates the interval relation between them
def calculate_relationship(a_s, a_e, b_s, b_e, set_name='window'):
    temp_distance = (np.sign(b_s - a_s), np.sign(b_e - a_e),
                     np.sign(b_s - a_e), np.sign(b_e - a_s))
    if set_name == 'window':
        return WINDOW_INTERVAL_RELATION_MAP.get(temp_distance, '')
    elif set_name == 'event':
        return EVENT_INTERVAL_RELATION_MAP.get(temp_distance, 0)
    else:
        return EVALUATION_INTERVAL_RELATION_MAP.get(temp_distance, 0)


# verifies if there is a valid interval relation between the event and the given window
def overlaps(s_time, e_time, td, event_name):
    s_label = event_name + "_s"
    e_label = event_name + "_e"

    if s_label in td and calculate_relationship(s_time, e_time, td[s_label], td[e_label]) != '':
        return True
    return False


# determines the predicted labels for the audio data
def label_data_aud(frame_size, stride, frame_num, sequence_len, td):
    predicted_label_data = np.zeros((BATCH_SIZE, AUD_CLASSES)).astype(float)
    aud_label = 0

    s_frame = stride * frame_num
    e_frame = s_frame + frame_size

    if e_frame > sequence_len:
        e_frame = sequence_len

    if overlaps(s_frame, e_frame, td, "command"):
        aud_label = 1
    if overlaps(s_frame, e_frame, td, "prompt"):
        aud_label = 1
    if overlaps(s_frame, e_frame, td, "reward"):
        aud_label = 1
    if overlaps(s_frame, e_frame, td, "abort"):
        aud_label = 1
    if overlaps(s_frame, e_frame, td, "noise_0"):
        aud_label = 1
    if overlaps(s_frame, e_frame, td, "noise_1"):
        aud_label = 1
    if overlaps(s_frame, e_frame, td, "audio_0"):
        aud_label = 2
    if overlaps(s_frame, e_frame, td, "audio_1"):
        aud_label = 2

    predicted_label_data[0][aud_label] = 1
    return predicted_label_data


# determines the predicted labels for the video data
def label_data_opt(frame_size, stride, frame_num, sequence_len, td):
    predicted_label_data = np.zeros((BATCH_SIZE, OPT_CLASSES)).astype(float)
    opt_label = 0

    s_frame = stride * frame_num
    e_frame = s_frame + frame_size

    if e_frame > sequence_len:
        e_frame = sequence_len

    if overlaps(s_frame, e_frame, td, "command"):
        opt_label = 1
    if overlaps(s_frame, e_frame, td, "prompt"):
        opt_label = 1
    if overlaps(s_frame, e_frame, td, "noise_0"):
        opt_label = 1
    if overlaps(s_frame, e_frame, td, "noise_1"):
        opt_label = 1
    if overlaps(s_frame, e_frame, td, "gesture_0"):
        opt_label = 2
    if overlaps(s_frame, e_frame, td, "gesture_1"):
        opt_label = 2

    predicted_label_data[0][opt_label] = 1
    return predicted_label_data


# prints the ground truth for the timing information
def process_real_times(td):
    final_td = dict()
    delete_prompt = False
    mapping = {'noise_0_s': 'command_s', 'noise_0_e': 'command_e',
               'noise_1_s': 'prompt_s', 'noise_1_e': 'prompt_e',
               'audio_0_s': 'response_s', 'audio_0_e': 'response_e',
               'audio_1_s': 'response_s', 'audio_1_e': 'response_e',
               'gesture_0_s': 'response_s', 'gesture_0_e': 'response_e',
               'gesture_1_s': 'response_s', 'gesture_1_e': 'response_e'}
    if td.get('audio_0_s', None) is not None and td.get('audio_1_s', None) is not None:
        del td['audio_1_s']
        del td['audio_1_e']
        td['reward_s'] = td['noise_1_s']
        td['reward_e'] = td['noise_1_e']
        delete_prompt = True
    if td.get('gesture_0_s', None) is not None and td.get('gesture_1_s', None) is not None:
        del td['gesture_1_s']
        del td['gesture_1_e']
        td['reward_s'] = td['noise_1_s']
        td['reward_e'] = td['noise_1_e']
        delete_prompt = True
    if delete_prompt:
        del td['prompt_s']
        del td['prompt_e']
        del td['noise_1_s']
        del td['noise_1_e']
    if td.get('reward_s', None) is not None:
        mapping['abort_s'] = 'reward_s'
        mapping['abort_e'] = 'reward_e'
    for event, time in td.items():
        event = mapping.get(event, event)
        event_name = event.replace('_s', '').replace('_e', '')
        curr_time = final_td.get(event_name, (100000, -1))
        if '_s' in event:
            new_time = (min(time, curr_time[0]), curr_time[1])
        else:
            new_time = (curr_time[0], max(time, curr_time[1]))
        final_td[event_name] = new_time
    # for event in sorted(final_td):
    #     print('{}: {}'.format(event, final_td[event]))
    # print('DEBUG: {}')
    # for event in sorted(td):
    #     print('{}: {}'.format(event, td[event]))
    return final_td


if __name__ == '__main__':
    print("time start: {}".format(datetime.now()))

    # MODELS PREPARATION ##############################################
    # read contents of TFRecord file and generate list of file names
    file_names = list()
    for root, directory, files in os.walk(TF_RECORDS_PATH):
        for f in files:
            if validation:
                if 'validation' in f:
                    file_names.append(os.path.join(root, f))
            else:
                if 'validation' not in f:
                    file_names.append(os.path.join(root, f))
    file_names.sort()
    print("{}".format(file_names))

    # loading itbn model
    nx_model = nx.read_gpickle(ITBN_MODEL_PATH)
    itbn_model = IntervalTemporalBayesianNetwork(nx_model.edges())
    itbn_model.add_cpds(*nx_model.cpds)
    itbn_model.learn_temporal_relationships_from_cpds()

    # load cnn models
    aud_dqn = aud_classifier.ClassifierModel(batch_size=BATCH_SIZE, learning_rate=ALPHA,
                                             filename=AUD_DQN_CHKPNT)
    opt_dqn = opt_classifier.ClassifierModel(batch_size=BATCH_SIZE, learning_rate=ALPHA,
                                             filename=OPT_DQN_CHKPNT)

    # prepare tf objects
    aud_coord = tf.train.Coordinator()
    opt_coord = tf.train.Coordinator()

    # read records from files into tensors
    seq_len_inp, opt_raw_inp, aud_raw_inp, timing_labels_inp, timing_values_inp, file_name = \
        input_pipeline(file_names)

    # initialize variables
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

    print("Num epochs: {}\nBatch Size: {}\nNum Files: {}".format(NUM_EPOCHS, BATCH_SIZE,
                                                                 len(file_names)))

    # confusion matrices
    aud_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    opt_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    # results
    results = {'prompt': [0, 0], 'response': [0, 0], 'abort': [0, 0],
               'reward': [0, 0], 'command': [0, 0]}

    num_files = len(file_names)
    counter = 0
    while len(file_names) > 0:
        # read a batch of tfrecords into np arrays
        seq_len, opt_raw, aud_raw, timing_labels, timing_values, name = opt_dqn.sess.run(
         [seq_len_inp, opt_raw_inp, aud_raw_inp, timing_labels_inp, timing_values_inp, file_name])

        if validation:
            name = name[0].replace('.txt', '_validation.tfrecord').replace(LABELS_PATH,
                                                                           TF_RECORDS_PATH)
        else:
            name = name[0].replace('.txt', '.tfrecord').replace(LABELS_PATH, TF_RECORDS_PATH)

        if name in file_names:
            # print debugging feedback
            file_names.remove(name)
            counter += 1
            print("processing {}/{}: {}".format(counter, num_files, name))

            # get timing information and calculate number of chunks
            timing_dict = parse_timing_dict(timing_labels[0], timing_values[0])
            aud_num_chunks = (seq_len - AUD_FRAME_SIZE) / AUD_STRIDE + 1
            opt_num_chunks = (seq_len - OPT_FRAME_SIZE) / OPT_STRIDE + 1

            # initialize control and debugging variables
            aud_chunk_counter = 0
            opt_chunk_counter = 0
            window_processed = False
            aud_selected_class = 0
            opt_selected_class = 0
            aud_real_sequence = ""
            aud_pred_sequence = ""
            opt_real_sequence = ""
            opt_pred_sequence = ""

            # initialize itbn status
            obs_robot = 0
            obs_human = 0
            session_data = pd.DataFrame([('N', 'N', 'N', 'N', 'N',
                                          obs_robot, obs_robot, obs_robot, obs_human, obs_robot,
                                          0, 0, 0, 0, 0)],
                                        columns=['abort', 'command', 'prompt', 'response', 'reward',
                                                 'obs_abort', 'obs_command', 'obs_prompt',
                                                 'obs_response', 'obs_reward', 'tm_command_prompt',
                                                 'tm_command_response', 'tm_prompt_abort',
                                                 'tm_prompt_response', 'tm_response_reward'])
            window_data = session_data.copy(deep=True)
            pending_events = ['abort', 'command', 'prompt', 'response', 'reward']
            robot_events = ['obs_abort', 'obs_command', 'obs_prompt', 'obs_reward']
            human_events = ['obs_response']
            start_event = 'command'
            terminal_events = ['abort', 'reward']
            events_priority = {1: 'reward', 2: 'abort', 3: 'prompt', 4: 'response'}
            terminate = False
            event_times = dict()
            w_time = (0, 0)
            last_obs = -1
            last_event = ''

            for i in range(seq_len):
                window_processed = False
                if i == AUD_STRIDE * aud_chunk_counter + AUD_FRAME_SIZE:
                    with aud_dqn.sess.as_default():
                        start_frame = AUD_STRIDE * aud_chunk_counter
                        end_frame = AUD_STRIDE * aud_chunk_counter + AUD_FRAME_SIZE
                        aud_label_data = label_data_aud(AUD_FRAME_SIZE, AUD_STRIDE,
                                                        aud_chunk_counter, seq_len, timing_dict)
                        vals = {
                            aud_dqn.seq_length_ph: seq_len,
                            aud_dqn.aud_ph: np.expand_dims(aud_raw[0][start_frame: end_frame], 0),
                            aud_dqn.aud_y_ph: aud_label_data
                        }
                        aud_pred = aud_dqn.sess.run([aud_dqn.aud_observed], feed_dict=vals)
                        real_class = int(np.argmax(aud_label_data))
                        aud_selected_class = int(aud_pred[0][0])
                        aud_matrix[real_class][aud_selected_class] += 1
                        aud_real_sequence += SEQUENCE_CHARS[real_class]
                        aud_pred_sequence += SEQUENCE_CHARS[aud_selected_class]
                        aud_chunk_counter += 1
                        window_processed = True
                        w_time = (start_frame, end_frame)
                if i == OPT_STRIDE * opt_chunk_counter + OPT_FRAME_SIZE:
                    with opt_dqn.sess.as_default():
                        start_frame = OPT_STRIDE * opt_chunk_counter
                        end_frame = OPT_STRIDE * opt_chunk_counter + OPT_FRAME_SIZE
                        opt_label_data = label_data_opt(OPT_FRAME_SIZE, OPT_STRIDE,
                                                        opt_chunk_counter, seq_len, timing_dict)
                        vals = {
                            opt_dqn.seq_length_ph: seq_len,
                            opt_dqn.pnt_ph: np.expand_dims(opt_raw[0][start_frame: end_frame], 0),
                            opt_dqn.pnt_y_ph: opt_label_data
                        }
                        opt_pred = opt_dqn.sess.run([opt_dqn.wave_observed], feed_dict=vals)
                        real_class = int(np.argmax(opt_label_data))
                        opt_selected_class = int(opt_pred[0][0])
                        opt_matrix[real_class][opt_selected_class] += 1
                        opt_real_sequence += SEQUENCE_CHARS[real_class]
                        opt_pred_sequence += SEQUENCE_CHARS[opt_selected_class]
                        opt_chunk_counter += 1
                        w_time = (start_frame, end_frame)
                if not terminate and window_processed:
                    if opt_selected_class == 2 and aud_selected_class == 0:
                        important_obs = opt_selected_class
                    else:
                        important_obs = aud_selected_class
                    obs_robot = 0
                    obs_human = 0
                    window_rels = dict()
                    if important_obs == 1:
                        obs_robot = 1
                    if important_obs == 2:
                        obs_human = 1
                    if start_event in pending_events and obs_robot == 1:
                        session_data[start_event][0] = 'Y'
                        pending_events.remove(start_event)
                        event_times[start_event] = w_time
                        last_obs = important_obs
                        last_event = start_event
                    elif start_event not in pending_events and important_obs != last_obs:
                        last_obs = important_obs
                        window_data = session_data.copy(deep=True)
                        for col in list(window_data.columns):
                            if col in robot_events:
                                window_data[col][0] = obs_robot
                            elif col in human_events:
                                window_data[col][0] = obs_human
                            elif col.startswith(itbn_model.temporal_node_marker):
                                events = col.split('_')
                                a_times = event_times.get(events[1], (FAR_FRAME, FAR_FRAME + 1))
                                rel = calculate_relationship(a_times[0], a_times[1], w_time[0],
                                                             w_time[1], set_name='event')
                                if rel not in itbn_model.relation_map[(events[1], events[2])]:
                                    rel = 0
                                window_rels[(events[1], events[2])] = rel
                        new_preds = list()
                        # print('FRAME: {}'.format(i))
                        for event in pending_events:
                            temp_window = window_data.copy(deep=True)
                            for events, rel in window_rels.items():
                                if event in events and temp_window['obs_' + event][0] == 1:
                                    temp_window[itbn_model.temporal_node_marker + events[0] + '_' +
                                                events[1]] = rel
                            temp_window.drop(event, axis=1, inplace=True)
                            predictions = itbn_model.predict(temp_window)
                            # print('prediction with {}, {}: {}'.format(
                            #     obs_robot, obs_human, dict(predictions.ix[0])))
                            if predictions[event][0] == 'Y':
                                new_preds.append(event)
                        if len(new_preds) == 0:
                            last_obs = 0
                        for event_code in sorted(events_priority):
                            event = events_priority[event_code]
                            if event in new_preds:
                                event_times[event] = w_time
                                last_event = event
                                session_data[event][0] = 'Y'
                                pending_events.remove(event)
                                for events, rel in window_rels.items():
                                    if event in events:
                                        session_data[itbn_model.temporal_node_marker + events[0] +
                                                     '_' + events[1]][0] = rel
                                if event in terminal_events:
                                    terminate = True
                                break
                        if terminate:
                            aud_pred_sequence += '  '
                            opt_pred_sequence += '  '
                    else:
                        if last_event != '':
                            curr_time = event_times[last_event]
                            new_time = (curr_time[0], w_time[1])
                            event_times[last_event] = new_time
            # print('REAL TIMES:')
            final_td = process_real_times(timing_dict)
            # print('PREDICTED TIMES:')
            error_detected = False
            for event in sorted(event_times):
                real_time = final_td.get(event, (-1, -1))
                correct = (event_times[event][0] <= real_time[0] <=
                           event_times[event][0] + AUD_FRAME_SIZE)
                if correct:
                    results[event][1] = results[event][1] + 1
                else:
                    error_detected = True
                    results[event][0] = results[event][0] + 1
                    print('{}: {} - {}, {}, {}'.format(
                        event, correct, event_times[event][0], real_time[0],
                        event_times[event][0] + AUD_FRAME_SIZE))
            if error_detected:
                # print debugging timing information
                print(aud_real_sequence + "\n" + aud_pred_sequence)
                print(opt_real_sequence + "\n" + opt_pred_sequence)
    # print confusion matrices
    print("time end: {}\nAUDIO\n{}\n\nVIDEO\n{}\n".format(datetime.now(), aud_matrix, opt_matrix))
    print('RESULTS: {}'.format(results))
