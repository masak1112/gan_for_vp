import importlib
import layer_def as ld
importlib.reload(ld)
from VideoPredictionBase import *
from BasicConvLSTMCell import *

class convLSTM(VideoPredictionBase):
    def __init__(self, data_params_path, model_hparams_path,output_params_path):
        # get configuration and model hparms
        super(convLSTM, self).__init__(data_params_path, model_hparams_path, output_params_path)
        # Build the graph
        self.build()
        # Initialize session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        # Summary op
        self.loss_summary = tf.summary.scalar("total_losses", self.total_loss)
        self.summary_op = tf.summary.merge_all()
        #build writer
        Path(self.checkpoint_dir).mkdir(parents = True, exist_ok=True)
        self.train_log_file = self.base_dir + "/train_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.val_log_file = self.base_dir + "/val_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_writer = tf.summary.FileWriter(self.train_log_file, self.sess.graph)
        self.val_writer = tf.summary.FileWriter(self.val_log_file, self.sess.graph)


    @staticmethod
    def convLSTM_cell(inputs, hidden):
        conv1 = ld.conv_layer(inputs, 3, 2, 8, "encode_1", activate = "leaky_relu")
        conv2 = ld.conv_layer(conv1, 3, 1, 8, "encode_2", activate = "leaky_relu")
        conv3 = ld.conv_layer(conv2, 3, 2, 8, "encode_3", activate = "leaky_relu")
        y_0 = conv3
        # conv lstm cell
        with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
            cell = BasicConvLSTMCell(shape = [16, 16], filter_size = [3, 3], num_features = 8)
            if hidden is None:
                hidden = cell.zero_state(y_0, tf.float32)
            output, hidden = cell(y_0, hidden)
        output_shape = output.get_shape().as_list()
        z3 = tf.reshape(output, [-1, output_shape[1], output_shape[2], output_shape[3]])
        conv5 = ld.transpose_conv_layer(z3, 3, 2, 8, "decode_5", activate = "leaky_relu")
        conv6 = ld.transpose_conv_layer(conv5, 3, 1, 8, "decode_6", activate = "leaky_relu")
        x_hat = ld.transpose_conv_layer(conv6, 3, 2, 3, "decode_7", activate = "sigmoid")  # set activation to linear
        return x_hat, hidden

    def convLSTM_network(self):
        # make the template to share the variables
        network_template = tf.make_template('network', convLSTM.convLSTM_cell)
        # create network
        x_hat_context = []
        x_hat_predict = []
        seq_start = 1
        hidden = None
        for i in range(self.context_frames):
            if i < seq_start:
                x_1, hidden = network_template(self.x[:, i, :, :, :], hidden)
            else:
                x_1, hidden = network_template(x_1, hidden)
            x_hat_context.append(x_1)

        for i in range(self.predict_frames):
            x_1, hidden = network_template(x_1, hidden)
            x_hat_predict.append(x_1)

        # pack them all together
        x_hat_context = tf.stack(x_hat_context)
        x_hat_predict = tf.stack(x_hat_predict)
        self.x_hat_context = tf.transpose(x_hat_context, [1, 0, 2, 3, 4])  # change first dim with sec dim
        self.x_hat_predict = tf.transpose(x_hat_predict, [1, 0, 2, 3, 4])  # change first dim with sec dim
        return self.x_hat_context, self.x_hat_predict

    # Build the netowrk and the loss functions
    def build(self):
        tf.reset_default_graph()
        tf.set_random_seed(12345)
        self.train_iterator = self.make_dataset(type = "train")
        self.val_iterator = self.make_dataset(type = "val")
        self.test_iterator = self.make_dataset(type = "test")
        self.x = tf.placeholder(tf.float32, [None, 20, 64, 64, 3])
        self.global_step = tf.train.get_or_create_global_step()

        self.x_hat_context_frames, self.x_hat_predict_frames = self.convLSTM_network()
        self.x_hat = tf.concat([self.x_hat_context_frames, self.x_hat_predict_frames], 1)

        # Loss calculation
        self.context_frames_loss = tf.reduce_mean(
            tf.square(self.x[:, :self.context_frames, :, :, 0] - self.x_hat_context_frames[:, :, :, :, 0]))
        self.predict_frames_loss = tf.reduce_mean(
            tf.square(self.x[:, self.context_frames:, :, :, 0] - self.x_hat_predict_frames[:, :, :, :, 0]))
        self.total_loss = self.context_frames_loss + self.predict_frames_loss
        self.train_op = tf.train.AdamOptimizer(
            learning_rate = self.lr).minimize(self.total_loss, global_step = self.global_step)
        # Build a saver
        self.saver = tf.train.Saver(tf.global_variables())
        return

    # Execute the forward and the backward pass
    def run_single_step(self, global_step):
        try:
            train_batch = self.sess.run(self.train_iterator.get_next())
            x_hat, train_summary, _, train_losses = self.sess.run(
                [self.x_hat, self.summary_op, self.train_op, self.total_loss],
                feed_dict = {self.x: train_batch["images"]})
            self.train_writer.add_summary(train_summary, global_step)

        except tf.errors.OutOfRangeError:
            print("train out of range error")
        try:
            val_batch = self.sess.run(self.val_iterator.get_next())
            val_summary, val_losses = self.sess.run([self.summary_op, self.total_loss],
                                                    feed_dict = {self.x: val_batch["images"]})
            self.val_writer.add_summary(val_summary, global_step)
        except tf.errors.OutOfRangeError:
            print("train out of range error")
        return train_losses, val_losses


    def train(self):
        # Training loop
        global_step = self.sess.run(self.global_step)
        self.sess.run(self.train_iterator.initializer)
        self.sess.run(self.val_iterator.initializer)
        for epoch in range(self.num_epochs):
            # Run an epoch
            for iter in range(self.num_sample // self.batch_size):
                print("iter", iter)
                global_step = self.sess.run(self.global_step)
                train_losses, val_losses = self.run_single_step(global_step)
                print("Train_loss: {}; Val_loss{} for global step {}".format(train_losses, val_losses, global_step))
                checkpoint_path = os.path.join(self.checkpoint_dir, self.model + '.ckpt')
                self.saver.save(self.sess, checkpoint_path)
        return None


    def restore_model(self):
        # restore the existing checkpoints
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        # Extract from checkpoint filename
        global_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        sess = tf.Session()
        print("Restore from {}".format(ckpt.model_checkpoint_path))
        # graph = tf.get_default_graph()
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, tf.train.latest_checkpoint(self.checkpoint_dir))
        loaded_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)



def main():
    data_params_path = "./hparams/data_hparams.json"
    model_hparams_path = "./hparams/model_hparams.json"
    output_base = "./hparams/output_params.json"
    exp = convLSTM(data_params_path, model_hparams_path, output_base)
    exp.train()

if __name__ == '__main__':
    main()