import numpy as np
import tensorflow as tf


class WideModel(object):
    def __init__(self, embed_size, user_size, item_size, lr,
                 optim, initializer, loss_func, activation_func,
                 regularizer_rate, iterator, topk, dropout, is_training):
        """
        Important Arguments.

        embed_size: The final embedding size for users and items.
        optim: The optimization method chosen in this model.
        initializer: The initialization method.
        loss_func: Loss function, we choose the cross entropy.
        regularizer_rate: L2 is chosen, this represents the L2 rate.
        iterator: Input dataset.
        topk: For evaluation, computing the topk items.
        """

        self.embed_size = embed_size
        self.user_size = user_size
        self.item_size = item_size
        self.lr = lr
        self.initializer = initializer
        self.loss_func = loss_func
        self.activation_func = activation_func
        self.regularizer_rate = regularizer_rate
        self.optim = optim
        self.topk = topk
        self.dropout = dropout
        self.is_training = is_training
        self.iterator = iterator

    def get_data(self):
        sample = self.iterator.get_next()
        self.user = sample['user']
        self.item = sample['item']
        self.label = tf.cast(sample['label'], tf.float32)

    def inference(self):
        """ Initialize important settings """
        # self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.regularizer_rate)
        self.regularizer = tf.constant(self.regularizer_rate, dtype=tf.float32, shape=[], name="l2")

        if self.initializer == 'Normal':
            self.initializer = tf.truncated_normal_initializer(stddev=0.01)
        elif self.initializer == 'Xavier_Normal':
            self.initializer = tf.contrib.layers.xavier_initializer()
        else:
            self.initializer = tf.glorot_uniform_initializer()

        if self.activation_func == 'ReLU':
            self.activation_func = tf.nn.relu
        elif self.activation_func == 'Leaky_ReLU':
            self.activation_func = tf.nn.leaky_relu
        elif self.activation_func == 'ELU':
            self.activation_func = tf.nn.elu

        if self.loss_func == 'cross_entropy':
            # self.loss_func = lambda labels, logits: -tf.reduce_sum(
            # 		(labels * tf.log(logits) + (
            # 		tf.ones_like(labels, dtype=tf.float32) - labels) *
            # 		tf.log(tf.ones_like(logits, dtype=tf.float32) - logits)), 1)
            self.loss_func = tf.nn.sigmoid_cross_entropy_with_logits

        if self.optim == 'SGD':
            self.optim = tf.train.GradientDescentOptimizer(self.lr,
                                                           name='SGD')
        elif self.optim == 'RMSProp':
            self.optim = tf.train.RMSPropOptimizer(self.lr, decay=0.9,
                                                   momentum=0.0, name='RMSProp')
        elif self.optim == 'Adam':
            self.optim = tf.train.AdamOptimizer(self.lr, name='Adam')

    def create_model(self):
        # 创建全局偏置
        bias_global = tf.Variable(tf.constant(0.01, shape=[1]))

        # 创建广度用户偏置
        w_bias_user_wide = tf.Variable(tf.constant(0.01, shape=[self.user_size]))
        # 创建广度物品偏置
        w_bias_item_wide = tf.Variable(tf.constant(0.01, shape=[self.item_size]))
        # 通过用户的id取出对应的用户偏置
        bias_user_wide = tf.nn.embedding_lookup(w_bias_user_wide, self.user)
        # 通过物品的id取出对应的物品偏置
        bias_item_wide = tf.nn.embedding_lookup(w_bias_item_wide, self.item)
        # 创建用户隐语义向量
        w_user_wide = tf.Variable(tf.truncated_normal(shape=[self.user_size, self.embed_size], stddev=0.02))
        # 创建物品隐语义向量
        w_item_wide = tf.Variable(tf.truncated_normal(shape=[self.item_size, self.embed_size], stddev=0.02))
        # 通过用户的id取出对应的用户隐语义
        embd_user_wide = tf.nn.embedding_lookup(w_user_wide, self.user)
        # 通过物品id取出对应的物品隐语义
        embd_item_wide = tf.nn.embedding_lookup(w_item_wide, self.item)

        # 得到宽度模型
        wide = tf.reduce_sum(tf.multiply(embd_user_wide, embd_item_wide), 1) + bias_user_wide + bias_item_wide
        # 加入宽度模型的正则项
        reg_wide = tf.add(tf.nn.l2_loss(embd_user_wide), tf.nn.l2_loss(embd_item_wide))

        self.logits = wide + bias_global
        self.logits_dense = tf.reshape(self.logits, [-1])

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(self.loss_func(
                labels=self.label, logits=self.logits_dense, name='loss'))
        with tf.name_scope("optimzation"):
            # cost = self.loss + tf.contrib.layers.apply_regularization(regularizer=self.regularizer, weights_list=[embd_item_wide, embd_item_wide])
            # self.regularizer(embd_item_wide)
            # self.regularizer(embd_user_wide)
            # cost = self.loss + tf.contrib.layers.apply_regularization(regularizer=self.regularizer)
            cost = self.loss + self.regularizer * reg_wide
            self.optimzer = self.optim.minimize(cost)

    def eval(self):
        with tf.name_scope("evaluation"):
            self.item_replica = self.item
            _, self.indice = tf.nn.top_k(tf.sigmoid(self.logits_dense), self.topk)

    def summary(self):
        """ Create summaries to write on tensorboard. """
        self.writer = tf.summary.FileWriter('./graphs/Wide', tf.get_default_graph())
        with tf.name_scope("summaries"):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def build(self):
        self.get_data()
        self.inference()
        self.create_model()
        self.eval()
        self.summary()
        self.saver = tf.train.Saver(tf.global_variables())

    def step(self, session, step):
        """ Train the model step by step. """
        if self.is_training:
            loss, optim, summaries = session.run(
                [self.loss, self.optimzer, self.summary_op])
            self.writer.add_summary(summaries, global_step=step)
        else:
            indice, item = session.run([self.indice, self.item_replica])
            prediction = np.take(item, indice)

            return prediction, item
