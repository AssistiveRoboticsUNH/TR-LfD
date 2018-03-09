#!/usr/bin/env python

# generate_itbn_tfrecords.py
# Madison Clark-Turner
# 12/2/2017

import heapq
import os

import rosbag

from itbn_classifier.common.itbn_tfrecord_rw import *
from itbn_classifier.tools.dqn_packager_itbn import *


topic_names = [
    '/action_finished',
    '/nao_robot/camera/top/camera/image_raw',
    '/nao_robot/microphone/naoqi_microphone/audio_raw'
]


def readTimingFile(filename):
    # generate a heap of timing event tuples
    ifile = open(filename, 'r')
    timing_queue = []
    line = ifile.readline()
    while (len(line) != 0):
        line = line.split()
        # tuple = (time of event, name of event)
        event_time = float(line[1])
        event_time = rospy.Duration(event_time)
        timing_queue.append((event_time, line[0]))
        line = ifile.readline()
    heapq.heapify(timing_queue)
    ifile.close()
    return timing_queue


def gen_TFRecord_from_file(out_dir, out_filename, bag_filename, timing_filename, flip=False):
    packager = DQNPackager(flip=flip)
    bag = rosbag.Bag(bag_filename)
    packager.p = False

    #######################
    ##   TIMING FILE    ##
    #######################

    # parse timing file
    timing_queue = readTimingFile(timing_filename)
    # get first timing event
    current_time = heapq.heappop(timing_queue)
    timing_dict = {}

    #######################
    ##     READ FILE     ##
    #######################

    all_timing_frames_found = False
    start_time = None

    for topic, msg, t in bag.read_messages(topics=topic_names):
        if (start_time == None):
            start_time = t

        if (not all_timing_frames_found and t > start_time + current_time[0]):
            # add the frame number anf timing label to frame dict
            timing_dict[current_time[1]] = packager.getFrameCount()
            if (len(timing_queue) > 0):
                current_time = heapq.heappop(timing_queue)
            else:
                all_timing_frames_found = True
        if (topic == topic_names[1]):
            packager.imgCallback(msg)
        elif (topic == topic_names[2]):
            packager.audCallback(msg)

    # perform data pre-processing steps
    packager.formatOutput()

    # print(timing_dict)

    # generate TFRecord data
    ex = make_sequence_example(
        packager.getImgStack(), img_dtype,
        packager.getPntStack(), pnt_dtype,
        packager.getAudStack(), aud_dtype,
        timing_dict,
        timing_filename)

    # write TFRecord data to file
    end_file = ".tfrecord"
    if (flip):
        end_file = "_flip" + end_file

    writer = tf.python_io.TFRecordWriter(out_dir + out_filename + end_file)
    writer.write(ex.SerializeToString())
    writer.close()

    packager.reset()
    bag.close()


if __name__ == '__main__':
    gen_single_file = True
    process_all_files = False
    view_single_file = True
    view_all_files = False

    rospy.init_node('gen_tfrecord', anonymous=True)

    #############################

    bagfile = os.environ["HOME"] + "/ITBN_bags/test_01/zga1.bag"
    timefile = os.environ["HOME"] + "/PycharmProjects/dbn_arl/labels/test_01/zga1.txt"
    timefile_dir = os.environ["HOME"] + "/PycharmProjects/dbn_arl/labels/"

    outfile = os.environ["HOME"] + "/ITBN_tfrecords/scrap.tfrecord"
    outdir = os.environ["HOME"] + "/ITBN_tfrecords/"

    #############################

    if (gen_single_file):
        # generate a single file and store it as a scrap.tfrecord; Used for Debugging
        outdir = os.environ["HOME"] + "/ITBN_bags_livetest/"
        timefile = os.environ["HOME"] + "/ITBN_bags_livetest/test.txt"
        bagfile = os.environ["HOME"] + "/ITBN_bags_livetest/g.bag"
        outfile = os.environ["HOME"] + "/ITBN_bags_livetest/g.tfrecord"
        gen_TFRecord_from_file(out_dir=outdir, out_filename="g", bag_filename=bagfile,
                               timing_filename=timefile, flip=False)

    outdir = os.environ["HOME"] + '/ITBN_tfrecords/'
    if process_all_files:
        path = os.environ["HOME"] + '/ITBN_bags/'
        for root, subFolders, files in os.walk(path):
            for f in files:
                if '.bag' in f:
                    out_path = root.replace(path, outdir) + '/'
                    tffile = f.replace('.bag', '')
                    bagfile = os.path.join(root, f)
                    timefile = bagfile.replace(path, timefile_dir).replace('.bag', '.txt')
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)
                    print('Generating: {}'.format(os.path.join(out_path, tffile)))
                    gen_TFRecord_from_file(out_dir=out_path, out_filename=tffile,
                                           bag_filename=bagfile,
                                           timing_filename=timefile, flip=False)

    file_set = set()
    if view_single_file:
        file_set.add(outfile)
    elif view_all_files:
        for root, subFolders, files in os.walk(outdir):
            for f in files:
                if '.tfrecord' in f:
                    file_set.add(os.path.join(root, f))
    for f in file_set:
        # read contents of scrap.tfrecord; Used for Debugging
        coord = tf.train.Coordinator()
        filename_queue = tf.train.string_input_producer([f])
        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            threads = tf.train.start_queue_runners(coord=coord)
            # parse TFrecord

            def processData(inp, data_type):
                data_s = tf.reshape(inp, [-1, data_type["cmp_h"], data_type["cmp_w"],
                                          data_type["num_c"]])
                return tf.cast(data_s, tf.uint8)

            context_parsed, sequence_parsed = parse_sequence_example(filename_queue)

            # information to extract from TFrecord
            seq_len = context_parsed["length"]
            # img_raw = processData(sequence_parsed["img_raw"], img_dtype)
            opt_raw = processData(sequence_parsed["opt_raw"], pnt_dtype)
            aud_raw = processData(sequence_parsed["aud_raw"], aud_dtype)
            timing_labels = context_parsed["timing_labels"]
            timing_values = sequence_parsed["timing_values"]
            name = context_parsed["example_id"]

            # extract data (set range to a val > 1 if multiple TFrecords stored in single file)
            for i in range(1):
                # l, i, p, a, tl, tv, n = sess.run([seq_len, img_raw, opt_raw, aud_raw, timing_labels, timing_values, name]) #<-- has img data
                l, p, a, tl, tv, n = sess.run(
                    [seq_len, opt_raw, aud_raw, timing_labels, timing_values, name])
                timing_dict = parse_timing_dict(tl, tv)
                # print(timing_dict)

            coord.request_stop()
            coord.join(threads)

            def show(data, d_type):
                tout = []
                out = []
                for i in range(data.shape[0]):
                    imf = np.reshape(data[i], (d_type["cmp_h"], d_type["cmp_w"], d_type["num_c"]))

                    limit_size = d_type["cmp_w"]
                    frame_limit = 12
                    if d_type["name"] == "aud":
                        frame_limit = 120

                    if (d_type["cmp_w"] > limit_size):
                        mod = limit_size / float(d_type["cmp_h"])
                        imf = cv2.resize(imf, None, fx=mod, fy=mod, interpolation=cv2.INTER_CUBIC)

                    if (imf.shape[2] == 2):
                        imf = np.concatenate((imf, np.zeros((d_type["cmp_h"], d_type["cmp_w"], 1))),
                                             axis=2)
                        imf[..., 0] = imf[..., 1]
                        imf[..., 2] = imf[..., 1]
                        imf = imf.astype(np.uint8)

                    if (i % frame_limit == 0 and i != 0):
                        if (len(tout) == 0):
                            tout = out.copy()
                        else:
                            tout = np.concatenate((tout, out), axis=0)
                        out = []
                    if (len(out) == 0):
                        out = imf
                    else:
                        out = np.concatenate((out, imf), axis=1)
                if (data.shape[0] % frame_limit != 0):
                    fill = np.zeros((d_type["cmp_h"], d_type["cmp_w"] * (frame_limit -
                                                                         (data.shape[0] % frame_limit)),
                                     d_type["num_c"]))  # .fill(255)
                    fill.fill(0)
                    out = np.concatenate((out, fill), axis=1)
                if (len(out) != 0):
                    if (len(tout) == 0):
                        tout = out.copy()
                    else:
                        tout = np.concatenate((tout, out), axis=0)
                    return tout

            # Use for visualizing Data Types
            show_from = 0
            p_img = show(p[show_from:], pnt_dtype)
            cv2.imwrite(f.replace('.tfrecord', '_p.jpg').replace('ITBN_bags_livetest', 'ITBN_bags_livetest'), p_img)
            # os.system('gnome-open test_p.jpg')

            a_img = show(a[show_from:], aud_dtype)
            cv2.imwrite(f.replace('.tfrecord', '_a.jpg').replace('ITBN_bags_livetest', 'ITBN_bags_livetest'), a_img)
            # os.system('gnome-open test_a.jpg')

            # img2 = show(i[show_from:], img_dtype)
            # cv2.imshow("pnt", img2)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()