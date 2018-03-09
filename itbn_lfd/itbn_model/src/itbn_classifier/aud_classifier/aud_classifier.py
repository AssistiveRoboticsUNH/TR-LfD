import tensorflow as tf

# contains information relating to input data size
from itbn_classifier.common.constants import *

# network layer information for A_CNN
aud_layer_elements = [-1, 16, 32, 128, AUD_CLASSES]
aud_output_sizes = [(32, 6), (16, 4), (4, 4)]
aud_filter_sizes = [(8, 3), (4, 3), (8, 3)]
aud_stride_sizes = [(4, 1), (2, 1), (4, 1)]
aud_padding_size = [(2, 0), (1, 0), (2, 1)]

'''
ClassifierModel generates q-values for a given input observation
'''


class ClassifierModel:
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
            self.variables_aud = {
                "W1": weight_variable("W_conv1_aud", [aud_filter_sizes[0][0],
                                                      aud_filter_sizes[0][1],
                                                      aud_dtype["num_c"],
                                                      aud_layer_elements[1]]),
                "b1": bias_variable("b_conv1_aud", [aud_layer_elements[1]]),
                "W2": weight_variable("W_conv2_aud", [aud_filter_sizes[1][0],
                                                      aud_filter_sizes[1][1],
                                                      aud_layer_elements[1],
                                                      aud_layer_elements[2]]),
                "b2": bias_variable("b_conv2_aud", [aud_layer_elements[2]]),
                "W3": weight_variable("W_conv3_aud", [aud_filter_sizes[2][0],
                                                      aud_filter_sizes[2][1],
                                                      aud_layer_elements[2],
                                                      aud_layer_elements[3]]),
                "b3": bias_variable("b_conv3_aud", [aud_layer_elements[3]]),
                "W_lstm": weight_variable("W_lstm", [aud_layer_elements[-2],
                                                     aud_layer_elements[-1]]),
                "b_lstm": bias_variable("b_lstm", [aud_layer_elements[-1]]),
                "W_fc": weight_variable("W_fc", [aud_layer_elements[-1] + 1,
                                                 aud_layer_elements[-1]]),
                "b_fc": bias_variable("b_fc", [aud_layer_elements[-1]])
            }

            # Placeholder variables
            # placeholder for the Audio data
            self.aud_ph = tf.placeholder("float",
                                         [self.__batch_size, None,
                                          aud_dtype["cmp_h"] * aud_dtype["cmp_w"] * aud_dtype["num_c"]],
                                         name="aud_placeholder")

            # placeholder for the sequence length
            self.seq_length_ph = tf.placeholder("int32", [self.__batch_size],
                                                name="seq_len_placeholder")

            # placeholder for the reward values to classify with
            self.aud_y_ph = tf.placeholder("float", [None, AUD_CLASSES], name="aud_y_placeholder")

            # Build Model Structure
            # initialize all variables in the network
            self.pred_aud_set = self.execute_aud_var_set()  # used to initialize variables

            # Q-value Generation Functions
            # return the action with the highest q-value
            self.aud_observed = tf.argmax(self.execute_aud(), 1)
            self.observe = tf.argmax(self.execute_aud(), 1)

            # Optimization Functions
            # get the difference between the q-values and the true output
            self.cross_entropy_aud = tf.nn.softmax_cross_entropy_with_logits(labels=self.aud_y_ph,
                                                                             logits=self.execute_aud())
            # optimize the network
            self.optimizer_aud = tf.train.AdamOptimizer(learning_rate=self.__alpha).minimize(
                self.cross_entropy_aud)

            # Evaluation Functions
            # return a boolean indicating whether the system correctly predicted the output
            self.correct_pred_aud = tf.equal(tf.argmax(self.aud_observed, 1),
                                             tf.argmax(self.aud_y_ph, 1))

            # the accuracy of the current batch
            self.accuracy_aud = tf.reduce_mean(tf.cast(self.correct_pred_aud, tf.float32))

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
    def execute_aud_var_set(self):
        # Initialize the model's structure
        return self.aud_model(
            self.seq_length_ph,
            self.aud_ph,
            tf.variable_scope("aud"),
            self.variables_aud
        )

    def execute_aud(self):
        # Generate the q-values of Q for the given input
        return self.aud_model(
            self.seq_length_ph,
            self.aud_ph,
            tf.variable_scope("aud", reuse=True),
            self.variables_aud
        )

    def gen_prediction(self, num_frames, aud_data, verbose=False):
        """
        Generate q-values for an input passed in as seperate data points. Used when
        by external systems (ROS) to run the model without having to import	tensorflow
            -num_frames: (int) the number of frames in the video
            -aud_data: (numpy array) an array that contains the audio data
            -verbose: (bool) print additional information
        """
        aud_pred = self.sess.run(self.observe, feed_dict={
            self.seq_length_ph: [num_frames],
            self.aud_ph: aud_data
        })

        if verbose:
            available_actions = ["PMT", "REW", "ABT"]
            print("Best action: " + available_actions[int(aud_pred[0])])

        return int(aud_pred[0])

    # The Model
    def process_vars(self, seq, data_type):
        # cast inputs to the correct data type
        seq_inp = tf.cast(seq, tf.float32)
        return tf.reshape(seq_inp, (self.__batch_size, -1, data_type["cmp_h"],
                                    data_type["cmp_w"], data_type["num_c"]))

    def check_legal_inputs(self, tensor, name):
        # ensure that the current tensor is finite (doesn't have any NaN values)
        return tf.verify_tensor_all_finite(tensor, "ERR: Tensor not finite - " + name, name=name)

    def aud_model(self, seq_length, aud_ph, variable_scope, var_aud):
        """
        -seq_length: (placeholder) the number of frames in the video
        -aud_ph: (placeholder) an array that contains the audio data
        -train_ph: (placeholder) a bool indicating whether the variables are being trained
        -variable_scope: (variable_scope) scope for the temporal data
        -var_aud: (dict) the variables for the audio input
        """
        # ---------------------------------------
        # Convolution Functions
        # ---------------------------------------
        def convolve_data_3layer_aud(input_data, variables, n, dtype):
            # pass data into through A_CNN
            def pad_tf(x, padding):
                return tf.pad(x, [[0, 0], [padding[0], padding[0]],
                                  [padding[1], padding[1]], [0, 0]], "CONSTANT")

            def gen_convolved_output(sequence, W, b, stride, num_hidden,
                                     new_size, padding='SAME'):
                conv = tf.nn.conv2d(sequence, W, strides=[1, stride[0], stride[1], 1],
                                    padding=padding) + b
                return tf.nn.relu(conv)

            input_data = tf.reshape(input_data,
                                    [-1, dtype["cmp_h"], dtype["cmp_w"], dtype["num_c"]],
                                    name=n + "_inp_reshape")

            for i in range(3):
                si = str(i + 1)

                input_data = pad_tf(input_data, aud_padding_size[i])
                padding = "VALID"

                input_data = gen_convolved_output(input_data, variables["W" + si],
                                                  variables["b" + si], aud_stride_sizes[i],
                                                  aud_layer_elements[i + 1], aud_output_sizes[i],
                                                  padding)
                input_data = self.check_legal_inputs(input_data, "conv" + si + "_" + n)

            return input_data

        # =======================================
        # Model Execution Begins Here
        # =======================================

        # CNN Stacks
        inp_data = self.process_vars(aud_ph, aud_dtype)
        conv_inp = convolve_data_3layer_aud(inp_data, var_aud, "aud", aud_dtype)
        conv_inp = tf.reshape(conv_inp, [self.__batch_size, -1,
                                         aud_output_sizes[-1][0] * aud_output_sizes[-1][0] *
                                         aud_layer_elements[-2]], name="combine_reshape_aud")

        # capture variables before changing scope
        W_lstm = var_aud["W_lstm"]
        b_lstm = var_aud["b_lstm"]

        with variable_scope as scope:
            # ---------------------------------------
            # Internal Temporal Information (LSTM)
            # ---------------------------------------
            # print("combined_data: ", combined_data.get_shape())
            lstm_cell = tf.contrib.rnn.LSTMCell(aud_layer_elements[-2],
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
