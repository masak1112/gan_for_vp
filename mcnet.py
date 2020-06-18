import importlib
import layer_def as ld
from VideoPredictionBase import *
from BasicConvLSTMCell import *
from mcnet_ops import *
from mcnet_utils import *



class MCNET(VideoPredictionBase):
    def __init__(self, data_params_path, model_hparams_path,output_params_path):
        # get configuration and model hparms
        super(MCNET, self).__init__(data_params_path, model_hparams_path, output_params_path)
        #intial specific for mcnet
        self.diff_shape = [self.batch_size, self.context_frames-1, self.height,
                           self.width, self.channel]
        self.is_train = True
        self.alpha = self.model_hparams_dict.alpha
        self.beta = self.model_hparams_dict.beta
        self.gf_dim = self.model_hparams_dict.gf_dim
        self.df_dim = self.model_hparams_dict.df_dim

        # Build the graph
        self.build()
        # Initialize session
        self.start_session()
        #build writer
        self.create_writer()


    # Build the netowrk and the loss functions
    def build(self):
        tf.reset_default_graph()
        tf.set_random_seed(12345)

        self.split_dataset() #prepare train,val and testing
        self.global_step = tf.train.get_or_create_global_step()
        self.mcnet_network()
        self.build_loss_functions()
        self.build_summary_op() #build summary operation
        self.calculate_trainable_vars()  # count trainable vars
        self.build_train_op() #build train operation
        # Build a saver
        self.saver = tf.train.Saver(tf.global_variables())
        return



    def mcnet_network(self):

        # self.xt = tf.placeholder(tf.float32, self.xt_shape, name='xt')
        self.xt = self.x[:, self.context_frames - 1, :, :, :]

        self.diff_in = tf.placeholder(tf.float32, self.diff_shape, name='diff_in')

        diff_in_all = []
        for t in range(1, self.context_frames):
            prev = self.x[:, t - 1:t, :, :, :]
            next = self.x[:, t:t+1, :, :, :]
            diff_in = tf.subtract(next, prev)
            diff_in_all.append(diff_in)

        self.diff_in = tf.concat(axis=1, values=diff_in_all)

        cell = BasicConvLSTMCell([self.height/ 8, self.width / 8], [3, 3], 256)

        pred = self.forward(self.diff_in, self.xt, cell)

        # Bing++++++++++
        # self.G = tf.concat(axis=3,values=pred)
        self.G = tf.concat(axis=1, values=pred)

        #combine groud truth and predict frames
        self.x_hat = tf.concat([self.x[:, :self.context_frames, :, :, :], self.G], 1)

        # Bing---------
        if self.is_train:
            # Bing+++++++++++++
            # self.true_sim = inverse_transform(self.target[:,:,:,SEF:,:])
            self.true_sim = self.x[:, self.context_frames:, :, :, :]
            # Bing--------------
            # Bing: the following make sure the channel is three dimension, if the channel is 1 then will be duplicated
            if self.channel==1: self.true_sim = tf.tile(self.true_sim, [1, 1, 1, 1, 3])
            # Bing+++++++++++++
            # Bing: the raw inputs shape is [batch_size, self.height,self.width, num_seq, channel]. tf.transpose will transpoe the shape into
            # [batch size*num_seq, image_size0, image_size1, channels], for our era5 case, we do not need transpose
            # self.true_sim = tf.reshape(tf.transpose(self.true_sim,[0,3,1,2,4]),
            #                             [-1, self.height,
            #                              self.width, 3])
            self.true_sim = tf.reshape(self.true_sim, [-1, self.height, self.width, 3])
            # Bing--------------

        # Bing+++++++++++++
        # self.gen_sim = inverse_transform(self.G)
        self.gen_sim = self.G

        if self.channel == 1: self.gen_sim = tf.tile(self.gen_sim, [1, 1, 1, 1, 3])
        # self.gen_sim = tf.reshape(tf.transpose(self.gen_sim,[0,3,1,2,4]),
        #                                [-1, self.height,
        #                                self.width, 3])

        self.gen_sim = tf.reshape(self.gen_sim, [-1, self.height, self.width, 3])

        # Bing+++++++++++++
        # Bing:the shape of the layer will be channels*num_seq, why ?
        # binput = tf.reshape(self.target[:,:,:,:self.context_frames,:],
        #                  [self.batch_size, self.height,
        #                   self.width, -1])
        binput = tf.reshape(tf.transpose(self.x[:, :self.context_frames, :, :, :], [0, 1, 2, 3, 4]),
                            [self.batch_size, self.height,
                             self.width, -1])
        # Bing--------------
        btarget = tf.reshape(tf.transpose(self.x[:, self.context_frames:, :, :, :], [0, 1, 2, 3, 4]),
                             [self.batch_size, self.height,
                              self.width, -1])
        bgen = tf.reshape(self.G, [self.batch_size,
                                   self.height,
                                   self.width, -1])

        good_data = tf.concat(axis = 3, values = [binput, btarget])
        gen_data = tf.concat(axis = 3, values = [binput, bgen])
        self.gen_data = gen_data

        with tf.variable_scope("DIS", reuse = False):
            self.D, self.D_logits = self.discriminator(good_data)

        with tf.variable_scope("DIS", reuse = True):
            self.D_, self.D_logits_ = self.discriminator(gen_data)

        self.outputs = {}
        self.outputs["gen_images"] = self.x_hat

        return None


    def build_loss_functions(self):
        self.L_p = tf.reduce_mean(
            tf.square(self.G - self.x[:, self.context_frames:, :, :, :]))

        self.L_gdl = gdl(self.gen_sim, self.true_sim, 1.)
        self.L_img = self.L_p + self.L_gdl

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = self.D_logits, labels = tf.ones_like(self.D)
            ))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = self.D_logits_, labels = tf.zeros_like(self.D_)
            ))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.L_GAN = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = self.D_logits_, labels = tf.ones_like(self.D_)
            ))
        self.total_loss = self.alpha * self.L_img + self.beta * self.L_GAN
        return None



    def build_summary_op(self):
        # Summary op
        self.loss_sum = tf.summary.scalar("L_img", self.L_img)
        self.L_p_sum = tf.summary.scalar("L_p", self.L_p)
        self.L_gdl_sum = tf.summary.scalar("L_gdl", self.L_gdl)
        self.L_GAN_sum = tf.summary.scalar("L_GAN", self.L_GAN)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        self.total_loss_sum = tf.summary.scalar("total_loss", self.total_loss)
        self.g_sum = tf.summary.merge([self.L_p_sum,
                                       self.L_gdl_sum, self.loss_sum,
                                       self.L_GAN_sum])
        self.d_sum = tf.summary.merge([self.d_loss_real_sum, self.d_loss_sum,
                                       self.d_loss_fake_sum])
        self.summary_op = tf.summary.merge_all()

        return None


    def calculate_trainable_vars(self):
        self.t_vars = tf.trainable_variables()
        self.g_vars = [var for var in self.t_vars if 'DIS' not in var.name]
        self.d_vars = [var for var in self.t_vars if 'DIS' in var.name]
        num_param = 0.0
        for var in self.g_vars:
            num_param += int(np.prod(var.get_shape()))
        print("Number of parameters: %d" % num_param)


    def build_train_op(self):
        # Training
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1 = 0.5).minimize(
            self.d_loss, var_list = self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1 = 0.5).minimize(
            self.alpha * self.L_img + self.beta * self.L_GAN, var_list = self.g_vars)
        self.train_op = [self.d_optim, self.g_optim]
        return None


    def forward(self, diff_in, xt, cell):
        # Initial state
        state = tf.zeros([self.batch_size, self.height / 8,
                          self.width / 8, 512])
        reuse = False
        # Encoder
        # Bing++++++++++++++++++++++++++++
        for t in range(self.context_frames - 1):
            enc_h, res_m = self.motion_enc(diff_in[:, t, :, :, :], reuse = reuse)
            h_dyn, state = cell(enc_h, state, scope = 'lstm', reuse = reuse)
            reuse = True
        pred = []
        # Decoder
        for t in range(self.predict_frames):
            if t == 0:
                h_cont, res_c = self.content_enc(xt, reuse = False)
                h_tp1 = self.comb_layers(h_dyn, h_cont, reuse = False)
                res_connect = self.residual(res_m, res_c, reuse = False)
                x_hat = self.dec_cnn(h_tp1, res_connect, reuse = False)
            else:
                enc_h, res_m = self.motion_enc(diff_in, reuse = True)
                h_dyn, state = cell(enc_h, state, scope = 'lstm', reuse = True)
                h_cont, res_c = self.content_enc(xt, reuse = reuse)
                h_tp1 = self.comb_layers(h_dyn, h_cont, reuse = True)
                res_connect = self.residual(res_m, res_c, reuse = True)
                x_hat = self.dec_cnn(h_tp1, res_connect, reuse = True)

            x_hat_gray = x_hat
            xt_gray = xt

            diff_in = x_hat_gray - xt_gray
            xt = x_hat
            # Bing++++++++++++++++++++++++++++
            # pred.append(tf.reshape(x_hat,[self.batch_size, self.height,
            #                        self.width, 1, self.channel]))
            pred.append(tf.reshape(x_hat, [self.batch_size, 1, self.height,
                                           self.width, self.channel]))
            # Bing----------------
        return pred

    def motion_enc(self, diff_in, reuse):
        res_in = []


        conv1 = relu(conv2d(diff_in, output_dim = self.gf_dim, k_h = 5, k_w = 5,
                            d_h = 1, d_w = 1, name = 'dyn1_conv1', reuse = reuse))
        res_in.append(conv1)
        pool1 = MaxPooling(conv1, [2, 2])

        conv2 = relu(conv2d(pool1, output_dim = self.gf_dim * 2, k_h = 5, k_w = 5,
                            d_h = 1, d_w = 1, name = 'dyn_conv2', reuse = reuse))
        res_in.append(conv2)
        pool2 = MaxPooling(conv2, [2, 2])

        conv3 = relu(conv2d(pool2, output_dim = self.gf_dim * 4, k_h = 7, k_w = 7,
                            d_h = 1, d_w = 1, name = 'dyn_conv3', reuse = reuse))
        res_in.append(conv3)
        pool3 = MaxPooling(conv3, [2, 2])
        return pool3, res_in

    def content_enc(self, xt, reuse):
        res_in = []
        conv1_1 = relu(conv2d(xt, output_dim = self.gf_dim, k_h = 3, k_w = 3,
                              d_h = 1, d_w = 1, name = 'cont_conv1_1', reuse = reuse))
        conv1_2 = relu(conv2d(conv1_1, output_dim = self.gf_dim, k_h = 3, k_w = 3,
                              d_h = 1, d_w = 1, name = 'cont_conv1_2', reuse = reuse))
        res_in.append(conv1_2)
        pool1 = MaxPooling(conv1_2, [2, 2])

        conv2_1 = relu(conv2d(pool1, output_dim = self.gf_dim * 2, k_h = 3, k_w = 3,
                              d_h = 1, d_w = 1, name = 'cont_conv2_1', reuse = reuse))
        conv2_2 = relu(conv2d(conv2_1, output_dim = self.gf_dim * 2, k_h = 3, k_w = 3,
                              d_h = 1, d_w = 1, name = 'cont_conv2_2', reuse = reuse))
        res_in.append(conv2_2)
        pool2 = MaxPooling(conv2_2, [2, 2])

        conv3_1 = relu(conv2d(pool2, output_dim = self.gf_dim * 4, k_h = 3, k_w = 3,
                              d_h = 1, d_w = 1, name = 'cont_conv3_1', reuse = reuse))
        conv3_2 = relu(conv2d(conv3_1, output_dim = self.gf_dim * 4, k_h = 3, k_w = 3,
                              d_h = 1, d_w = 1, name = 'cont_conv3_2', reuse = reuse))
        conv3_3 = relu(conv2d(conv3_2, output_dim = self.gf_dim * 4, k_h = 3, k_w = 3,
                              d_h = 1, d_w = 1, name = 'cont_conv3_3', reuse = reuse))
        res_in.append(conv3_3)
        pool3 = MaxPooling(conv3_3, [2, 2])
        return pool3, res_in

    def comb_layers(self, h_dyn, h_cont, reuse=False):
        comb1 = relu(conv2d(tf.concat(axis = 3, values = [h_dyn, h_cont]),
                            output_dim = self.gf_dim * 4, k_h = 3, k_w = 3,
                            d_h = 1, d_w = 1, name = 'comb1', reuse = reuse))
        comb2 = relu(conv2d(comb1, output_dim = self.gf_dim * 2, k_h = 3, k_w = 3,
                            d_h = 1, d_w = 1, name = 'comb2', reuse = reuse))
        h_comb = relu(conv2d(comb2, output_dim = self.gf_dim * 4, k_h = 3, k_w = 3,
                             d_h = 1, d_w = 1, name = 'h_comb', reuse = reuse))
        return h_comb

    def residual(self, input_dyn, input_cont, reuse=False):
        n_layers = len(input_dyn)
        res_out = []
        for l in range(n_layers):
            input_ = tf.concat(axis = 3, values = [input_dyn[l], input_cont[l]])
            out_dim = input_cont[l].get_shape()[3]
            res1 = relu(conv2d(input_, output_dim = out_dim,
                               k_h = 3, k_w = 3, d_h = 1, d_w = 1,
                               name = 'res' + str(l) + '_1', reuse = reuse))
            res2 = conv2d(res1, output_dim = out_dim, k_h = 3, k_w = 3,
                          d_h = 1, d_w = 1, name = 'res' + str(l) + '_2', reuse = reuse)
            res_out.append(res2)
        return res_out

    def dec_cnn(self, h_comb, res_connect, reuse=False):

        shapel3 = [self.batch_size, int(self.height / 4),
                   int(self.width / 4), self.gf_dim * 4]
        shapeout3 = [self.batch_size, int(self.height / 4),
                     int(self.width / 4), self.gf_dim * 2]
        depool3 = FixedUnPooling(h_comb, [2, 2])
        deconv3_3 = relu(deconv2d(relu(tf.add(depool3, res_connect[2])),
                                  output_shape = shapel3, k_h = 3, k_w = 3,
                                  d_h = 1, d_w = 1, name = 'dec_deconv3_3', reuse = reuse))
        deconv3_2 = relu(deconv2d(deconv3_3, output_shape = shapel3, k_h = 3, k_w = 3,
                                  d_h = 1, d_w = 1, name = 'dec_deconv3_2', reuse = reuse))
        deconv3_1 = relu(deconv2d(deconv3_2, output_shape = shapeout3, k_h = 3, k_w = 3,
                                  d_h = 1, d_w = 1, name = 'dec_deconv3_1', reuse = reuse))

        shapel2 = [self.batch_size, int(self.height / 2),
                   int(self.width / 2), self.gf_dim * 2]
        shapeout3 = [self.batch_size, int(self.height / 2),
                     int(self.width / 2), self.gf_dim]
        depool2 = FixedUnPooling(deconv3_1, [2, 2])
        deconv2_2 = relu(deconv2d(relu(tf.add(depool2, res_connect[1])),
                                  output_shape = shapel2, k_h = 3, k_w = 3,
                                  d_h = 1, d_w = 1, name = 'dec_deconv2_2', reuse = reuse))
        deconv2_1 = relu(deconv2d(deconv2_2, output_shape = shapeout3, k_h = 3, k_w = 3,
                                  d_h = 1, d_w = 1, name = 'dec_deconv2_1', reuse = reuse))

        shapel1 = [self.batch_size, self.height,
                   self.width, self.gf_dim]
        shapeout1 = [self.batch_size, self.height,
                     self.width, self.channel]
        depool1 = FixedUnPooling(deconv2_1, [2, 2])
        deconv1_2 = relu(deconv2d(relu(tf.add(depool1, res_connect[0])),
                                  output_shape = shapel1, k_h = 3, k_w = 3, d_h = 1, d_w = 1,
                                  name = 'dec_deconv1_2', reuse = reuse))
        xtp1 = tanh(deconv2d(deconv1_2, output_shape = shapeout1, k_h = 3, k_w = 3,
                             d_h = 1, d_w = 1, name = 'dec_deconv1_1', reuse = reuse))
        return xtp1

    def discriminator(self, image):
        h0 = lrelu(conv2d(image, self.df_dim, name = 'dis_h0_conv'))
        h1 = lrelu(batch_norm(conv2d(h0, self.df_dim * 2, name = 'dis_h1_conv'),
                              "bn1"))
        h2 = lrelu(batch_norm(conv2d(h1, self.df_dim * 4, name = 'dis_h2_conv'),
                              "bn2"))
        h3 = lrelu(batch_norm(conv2d(h2, self.df_dim * 8, name = 'dis_h3_conv'),
                              "bn3"))
        h = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'dis_h3_lin')

        return tf.nn.sigmoid(h), h




    def run_single_step(self):
        try:
            train_batch = self.sess.run(self.train_iterator.get_next())
            x = self.sess.run([self.x], feed_dict = {self.x: train_batch["images"]})
            global_step, _, g_sum = self.sess.run([self.global_step,self.g_optim, self.g_sum], feed_dict = {self.x: train_batch["images"]})
            _, d_sum = self.sess.run([self.d_optim, self.d_sum], feed_dict = {self.x: train_batch["images"]})

            gen_data, train_total_loss = self.sess.run([self.gen_data, self.total_loss],
                                                       feed_dict = {self.x: train_batch["images"]})

        except tf.errors.OutOfRangeError:
            print("train out of range error")

        try:
            val_batch = self.sess.run(self.val_iterator.get_next())
            val_total_loss = self.sess.run([self.total_loss], feed_dict = {self.x: val_batch["images"]})
            # self.val_writer.add_summary(val_summary, global_step)
        except tf.errors.OutOfRangeError:
            print("train out of range error")

        return train_total_loss, val_total_loss, global_step


    def train(self):
        # Training loop
        #global_step = self.sess.run(self.global_step)
        self.sess.run(self.train_iterator.initializer)
        self.sess.run(self.val_iterator.initializer)
        for epoch in range(self.num_epochs):
            # Run an epoch
            for iter in range(self.num_sample // self.batch_size):
                print("iter", iter)
                train_losses, val_losses, global_step = self.run_single_step()
                print("Train_loss: {}; Val_loss{} for global step {}".format(train_losses, val_losses, global_step))
                checkpoint_path = os.path.join(self.checkpoint_dir, self.model + '.ckpt')
                self.saver.save(self.sess, checkpoint_path)
        return None


def main():
    data_params_path = "./hparams/data_hparams.json"
    model_hparams_path = "./hparams/model_hparams.json"
    output_base = "./hparams/output_params.json"
    exp = MCNET(data_params_path, model_hparams_path, output_base)
    exp.train()

if __name__ == '__main__':
    main()