import tensorflow as tf

# contains information relating to input data size
from itbn_classifier.common.constants import *

# network layer information for P_CNN
layer_elements = [-1, 16, 32, 128, OPT_CLASSES]
output_sizes = [32, 16, 4]
filter_sizes = [4, 4, 8]
stride_sizes = [2, 2, 4]
padding_size = [1, 1, 2]

'''
ClassifierModel generates q-values for a given input observation
'''


class ClassifierModel:
    # Constructor
    """
    batch_size - int (1 by default)
    filename - string, location of file with saved model parameters (no model listed by default)
    learning_rate - float, speed at which the model trains (1e-5 by default)
    """
    def __init__(self, batch_size=1, filename="", learning_rate=1e-5):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.__batch_size = batch_size
            self.__alpha = learning_rate

            # Model variables
            def weight_variable(name, shape):
                initial = tf.truncated_normal(shape, stddev=0.1)
                return tf.Variable(initial, name=name)

            def bias_variable(name, shape):
                initial = tf.constant(0.1, shape=shape)
                return tf.Variable(initial, name=name)

            # Q variables
            self.variables_pnt = {
                "W1": weight_variable("W_conv1_pnt", [filter_sizes[0], filter_sizes[0],
                                                      pnt_dtype["num_c"], layer_elements[1]]),
                "b1": bias_variable("b_conv1_pnt", [layer_elements[1]]),
                "W2": weight_variable("W_conv2_pnt", [filter_sizes[1], filter_sizes[1],
                                                      layer_elements[1], layer_elements[2]]),
                "b2": bias_variable("b_conv2_pnt", [layer_elements[2]]),
                "W3": weight_variable("W_conv3_pnt", [filter_sizes[2], filter_sizes[2],
                                                      layer_elements[2], layer_elements[-2]]),
                "b3": bias_variable("b_conv3_pnt", [layer_elements[-2]]),
                "W_lstm": weight_variable("W_lstm", [layer_elements[-2], layer_elements[-1]]),
                "b_lstm": bias_variable("b_lstm", [layer_elements[-1]]),
                "W_fc": weight_variable("W_fc", [layer_elements[-1] + 1, layer_elements[-1]]),
                "b_fc": bias_variable("b_fc", [layer_elements[-1]])
            }

            # Placeholder variables
            # placeholder for the Optical Flow data
            self.pnt_ph = tf.placeholder("float",
                                         [self.__batch_size, None,
                                          pnt_dtype["cmp_h"] * pnt_dtype["cmp_w"] * pnt_dtype["num_c"]],
                                         name="pnt_placeholder")

            # placeholder for the sequence length
            self.seq_length_ph = tf.placeholder("int32", [self.__batch_size],
                                                name="seq_len_placeholder")

            # placeholder for the reward values to classify with
            self.pnt_y_ph = tf.placeholder("float", [None, OPT_CLASSES], name="pnt_y_placeholder")

            # Build Model Structure
            # initialize all variables in the network
            self.pred_wave_set = self.execute_wave_var_set()  # used to initialize variables

            # Q-value Generation Functions
            # return the action with the highest q-value
            self.wave_observed = tf.argmax(self.execute_wave(), 1)
            self.observe = tf.argmax(self.execute_wave(), 1)

            # Optimization Functions
            # get the difference between the q-values and the true output
            self.cross_entropy_wave = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.pnt_y_ph, logits=self.execute_wave())
            # optimize the network
            self.optimizer_wave = tf.train.AdamOptimizer(learning_rate=self.__alpha).minimize(
                self.cross_entropy_wave)

            # Evaluation Functions
            # return a boolean indicating whether the system correctly predicted the output
            self.correct_pred_wave = tf.equal(tf.argmax(self.wave_observed, 1),
                                              tf.argmax(self.pnt_y_ph, 1))

            # the accuracy of the current batch
            self.accuracy_wave = tf.reduce_mean(tf.cast(self.correct_pred_wave, tf.float32))

        # Initialization
        # Generate Session
        self.sess = tf.InteractiveSession(graph=self.graph)

        # Variable for generating a save checkpoint
        self.saver = tf.train.Saver()

        if len(filename) == 0:
            # initialize all model variables
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
            print("VARIABLE VALUES INITIALIZED")
        else:
            # restore variables from a checkpoint
            self.saver.restore(self.sess, filename)
            print("VARIABLE VALUES RESTORED FROM: " + filename)

    # Helper Functions
    def save_model(self, name="model.ckpt", save_dir=""):
        """
            save the model to a checkpoint file
            -name: (string) name of the checkpoint file
            -save_dir: (string) directory to save the file into
        """
        self.saver.save(self.sess, save_dir + '/' + name)

    # Executor Functions
    def execute_wave_var_set(self):
        # Initialize the model's structure
        return self.wave_model(
            self.seq_length_ph,
            self.pnt_ph,
            tf.variable_scope("wave"),
            tf.variable_scope("wave"),
            self.variables_pnt
        )

    def execute_wave(self):
        # Generate the q-values of Q for the given input
        return self.wave_model(
            self.seq_length_ph,
            self.pnt_ph,
            tf.variable_scope("wave"),
            tf.variable_scope("wave", reuse=True),
            self.variables_pnt
        )

    def gen_prediction(self, num_frames, opt_data, verbose=False):
        """
        Generate q-values for an input passed in as seperate data points. Used when
        by external systems (ROS) to run the model without having to import	tensorflow
            -num_frames: (int) the number of frames in the video
            -opt_data: (numpy array) an array that contains the optical data
            -verbose: (bool) print additional information
        """
        opt_pred = self.sess.run(self.observe, feed_dict={
            self.seq_length_ph: [num_frames],
            self.pnt_ph: opt_data
        })

        if verbose:
            available_actions = ["PMT", "REW", "ABT"]
            print("Best action: " + available_actions[int(opt_pred[0])])

        return int(opt_pred[0])

    # The Model
    def process_vars(self, seq, data_type):
        # cast inputs to the correct data type
        seq_inp = tf.cast(seq, tf.float32)
        return tf.reshape(seq_inp, (self.__batch_size, -1, data_type["cmp_h"],
                                    data_type["cmp_w"], data_type["num_c"]))

    def check_legal_inputs(self, tensor, name):
        # ensure that the current tensor is finite (doesn't have any NaN values)
        return tf.verify_tensor_all_finite(tensor, "ERR: Tensor not finite - " + name, name=name)

    def wave_model(self, seq_length, pnt_ph, variable_scope, variable_scope2, var_pnt):
        """
        -seq_length: (placeholder) the number of frames in the video
        -pnt_ph: (placeholder) an array that contains the optical flow data
        -train_ph: (placeholder) a bool indicating whether the variables are being trained
        -variable_scope: (variable_scope) scope for the CNN stacks
        -variable_scope2: (variable_scope) scope for the temporal data
        -var_pnt: (dict) the variables for the optical flow input
        """
        # Convolution Functions
        def convolve_data_3layer_pnt(input_data, variables, n, dtype):
            # pass data into through P_CNN
            def pad_tf(x, p):
                return tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "CONSTANT")

            def gen_convolved_output(sequence, W, b, stride, num_hidden, new_size, padding='SAME'):
                conv = tf.nn.conv2d(sequence, W, strides=[1, stride, stride, 1],
                                    padding=padding) + b
                return tf.nn.relu(conv)

            input_data = tf.reshape(input_data,
                                    [-1, dtype["cmp_h"], dtype["cmp_w"], dtype["num_c"]],
                                    name=n + "_inp_reshape")

            for i in range(3):
                si = str(i + 1)

                input_data = pad_tf(input_data, padding_size[i])
                padding = "VALID"

                input_data = gen_convolved_output(input_data, variables["W" + si],
                                                  variables["b" + si], stride_sizes[i],
                                                  layer_elements[i + 1], output_sizes[i], padding)
                input_data = self.check_legal_inputs(input_data, "conv" + si + "_" + n)

            return input_data

        # =======================================
        # Model Execution Begins Here
        # =======================================

        # CNN Stacks
        # Inception Network (INRV2)
        with variable_scope as scope:
            # P_CNN
            inp_data = self.process_vars(pnt_ph, pnt_dtype)
            conv_inp = convolve_data_3layer_pnt(inp_data, var_pnt, "pnt", pnt_dtype)
            conv_inp = tf.reshape(conv_inp, [self.__batch_size, -1,
                                             output_sizes[-1] * output_sizes[-1] *
                                             layer_elements[-2]], name="combine_reshape")

            # capture variables before changing scope
            W_lstm = var_pnt["W_lstm"]
            b_lstm = var_pnt["b_lstm"]

        with variable_scope2 as scope:
            # Internal Temporal Information (LSTM)
            lstm_cell = tf.contrib.rnn.LSTMCell(layer_elements[-2],
                                                use_peepholes=False,
                                                cell_clip=None,
                                                initializer=None,
                                                num_proj=None,
                                                proj_clip=None,
                                                forget_bias=1.0,
                                                state_is_tuple=True,
                                                activation=None,
                                                reuse=None
                                                )

            lstm_mat, _ = tf.nn.dynamic_rnn(
                cell=lstm_cell,
                inputs=conv_inp,
                dtype=tf.float32,
                sequence_length=seq_length,
                time_major=False
            )

            # if lstm_out is NaN replace with 0 to prevent model breakage
            lstm_mat = tf.where(tf.is_nan(lstm_mat), tf.zeros_like(lstm_mat), lstm_mat)
            lstm_mat = self.check_legal_inputs(lstm_mat, "lstm_mat")

            # extract relevant information from LSTM output using partitions
            lstm_out = tf.expand_dims(lstm_mat[0, -1], 0)

            # FC1
            fc1_out = tf.matmul(lstm_out, W_lstm) + b_lstm
            fc1_out = self.check_legal_inputs(fc1_out, "fc1")

            return fc1_out


if __name__ == '__main__':
    dqn = ClassifierModel()
