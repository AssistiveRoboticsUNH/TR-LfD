#!/usr/bin/env python

# Madison Clark-Turner
# 10/14/2017


import rospy
import time
import tensorflow as tf
from std_msgs.msg import Bool
from itbn_classifier.common.constants import *
from itbn_classifier.aud_classifier import aud_classifier
from itbn_classifier.opt_classifier import opt_classifier
from itbn_classifier.tools.dqn_packager_itbn import DQNPackager
from itbn_lfd.srv import *

# itbn model path
ITBN_MODEL_PATH = '../input/itbn.nx'

# cnn models paths
AUD_DQN_CHKPNT = "itbn_classifier/aud_classifier/itbn_aud_final/model.ckpt"
OPT_DQN_CHKPNT = "itbn_classifier/opt_classifier/itbn_opt_final/model.ckpt"

# cnn parameters
ALPHA = 1e-5
AUD_FRAME_SIZE = 20
AUD_STRIDE = 7
OPT_FRAME_SIZE = 20
OPT_STRIDE = 7

packager = None


def get_next_action(message):
    print("Executing get_next_action")
    next_act = -1
    if message.last_act == 0:
        packager.reset(hard_reset=True)
    while next_act <= message.last_act:
        next_act = packager.getNextAction()
    print("Action: {}".format(next_act))
    if next_act > 1:
        packager.reset(hard_reset=True)
    return ITBNGetNextActionResponse(next_act)


def start_server(service_name, srv, func):
    rospy.Service(service_name, srv, func)
    print("Service {} is ready.".format(service_name))
    rospy.spin()


if __name__ == '__main__':
    rospy.init_node("itbn_action_selector")
    pub_ready = rospy.Publisher("/itbn/ready", Bool, queue_size=10)

    aud_model = aud_classifier.ClassifierModel(batch_size=BATCH_SIZE, learning_rate=ALPHA,
                                             filename=AUD_DQN_CHKPNT)
    opt_model = opt_classifier.ClassifierModel(batch_size=BATCH_SIZE, learning_rate=ALPHA,
                                             filename=OPT_DQN_CHKPNT)

    aud_coord = tf.train.Coordinator()
    opt_coord = tf.train.Coordinator()

    # initialize variables
    with aud_model.sess.as_default():
        with aud_model.graph.as_default():
            aud_model.sess.run(tf.local_variables_initializer())
            aud_model.sess.graph.finalize()
            threads = tf.train.start_queue_runners(coord=aud_coord, sess=aud_model.sess)

    with opt_model.sess.as_default():
        with opt_model.graph.as_default():
            opt_model.sess.run(tf.local_variables_initializer())
            opt_model.sess.graph.finalize()
            threads = tf.train.start_queue_runners(coord=opt_coord, sess=opt_model.sess)

    pub_ready.publish(Bool(True))
    print("ITBN Model ready")
    packager = DQNPackager(aud_model, opt_model, AUD_FRAME_SIZE,
                           AUD_STRIDE, OPT_FRAME_SIZE, OPT_STRIDE)
    print("ITBN Packager ready")

    start_server("get_next_action", ITBNGetNextAction, get_next_action)
