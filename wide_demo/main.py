import time
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

from wide_model import WideModel
import os

GENRES = [
    'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', "IMAX", 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

if not os.path.exists(os.path.join(os.getcwd(), "data")):
    os.chdir("..")


class Metrics(object):
    @classmethod
    def mrr(cls, gt_item, pred_items):
        if gt_item in pred_items:
            index = np.where(pred_items == gt_item)[0][0]
            return np.reciprocal(float(index + 1))
        else:
            return 0

    @classmethod
    def hit(cls, gt_item, pred_items):
        if gt_item in pred_items:
            return 1
        return 0

    @classmethod
    def ndcg(cls, gt_item, pred_items):
        if gt_item in pred_items:
            index = np.where(pred_items == gt_item)[0][0]
            return np.reciprocal(np.log2(index + 2))
        return 0


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 1024, 'size of mini-batch.')
tf.app.flags.DEFINE_integer('test_neg', 3705, 'number of negative samples for test.')
tf.app.flags.DEFINE_integer('embedding_size', 16, 'the size for embedding user and item.')
tf.app.flags.DEFINE_integer('epochs', 20, 'the number of epochs.')
tf.app.flags.DEFINE_integer('topK', 10, 'topk for evaluation.')
tf.app.flags.DEFINE_string('optim', 'Adam', 'the optimization method.')
tf.app.flags.DEFINE_string('initializer', 'Xavier', 'the initializer method.')
tf.app.flags.DEFINE_string('loss_func', 'cross_entropy', 'the loss function.')
tf.app.flags.DEFINE_string('activation', 'ReLU', 'the activation function.')
tf.app.flags.DEFINE_string('model_dir', 'model/', 'the dir for saving model.')
tf.app.flags.DEFINE_float('regularizer', 0.0, 'the regularizer rate.')
tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate.')
tf.app.flags.DEFINE_float('dropout', 0.0, 'dropout rate.')


def train(train_data,test_data,user_size,item_size):
    with tf.Session() as sess:
        iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                   train_data.output_shapes)

        model = WideModel(FLAGS.embedding_size, user_size, item_size, FLAGS.lr,
                        FLAGS.optim, FLAGS.initializer, FLAGS.loss_func, FLAGS.activation,
                        FLAGS.regularizer, iterator, FLAGS.topK, FLAGS.dropout, is_training=True)

        model.build()

        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt:
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Creating model with fresh parameters.")
            sess.run(tf.global_variables_initializer())
        count = 0
        for epoch in range(FLAGS.epochs):
            sess.run(model.iterator.make_initializer(train_data))
            model.is_training = True
            model.get_data()
            start_time = time.time()

            try:
                while True:
                    model.step(sess, count)
                    count += 1
            except tf.errors.OutOfRangeError:
                print("Epoch %d training " % epoch + "Took: " + time.strftime("%H: %M: %S",
                                                                              time.gmtime(time.time() - start_time)))

            sess.run(model.iterator.make_initializer(test_data))
            model.is_training = False
            model.get_data()
            start_time = time.time()
            HR,MRR,NDCG = [],[],[]
            prediction, label = model.step(sess, None)
            try:
                while True:
                    prediction, label = model.step(sess, None)

                    label = int(label[0])
                    HR.append(Metrics.hit(label, prediction))
                    MRR.append(Metrics.mrr(label, prediction))
                    NDCG.append(Metrics.ndcg(label, prediction))
            except tf.errors.OutOfRangeError:
                hr = np.array(HR).mean()
                mrr = np.array(MRR).mean()
                ndcg = np.array(NDCG).mean()
                print("Epoch %d testing  " % epoch + "Took: " + time.strftime("%H: %M: %S",
                                                                              time.gmtime(time.time() - start_time)))
                print("HR is %.3f, MRR is %.3f, NDCG is %.3f" % (hr, mrr, ndcg))

        ################################## SAVE MODEL ################################
        checkpoint_path = os.path.join(FLAGS.model_dir, "NCF.ckpt")
        model.saver.save(sess, checkpoint_path)


class DataSet(object):
    user_size = 6040
    item_size = 3706

    # @classmethod
    # def read_user_bought(cls, path):
    #     df = pd.read_csv(path)
    #     df["items"] = df["items"].apply(lambda entry: [int(d) for d in entry[1:-1].split(',')])
    #     user_id = df["user_id"].values
    #     items = df["items"].values
    #     data_dic = {}
    #     for id, user in enumerate(user_id):
    #         data_dic[user] = []
    #         for index, item in enumerate(items[id]):
    #             data_dic[user].append(item)
    #     return data_dic
    #
    # @classmethod
    # def _add_negative(cls, features, user_negative, labels, numbers, is_training):
    #     feature_user, feature_item, labels_add, feature_dict = [], [], [], {}
    #
    #     for i in range(len(features)):
    #         user = features['user_id'][i]
    #         item = features['item_id'][i]
    #         label = labels[i]
    #
    #         feature_user.append(user)
    #         feature_item.append(item)
    #         labels_add.append(label)
    #
    #         if is_training:
    #             negative = user_negative[user][:]
    #         else:
    #             negative = set(user_negative[user][:])
    #             if item in negative:
    #                 negative.remove(item)
    #             negative = list(negative)
    #
    #         if numbers < len(negative):
    #             neg_samples = negative[:numbers]
                # neg_samples = np.random.choice(user_negative[user], size=numbers, replace=False).tolist()
            # else:
            #     neg_samples = negative
                # neg_samples = np.random.choice(user_negative[user], size=numbers).tolist()

            # if is_training:
            #     for k in neg_samples:
            #         feature_user.append(user)
            #         feature_item.append(k)
            #         labels_add.append(0)
            #
            # else:
            #     for k in neg_samples:
            #         feature_user.append(user)
            #         feature_item.append(k)
            #         labels_add.append(k)
        #
        # feature_dict['user'] = feature_user
        # feature_dict['item'] = feature_item
        #
        # return feature_dict, labels_add

    @classmethod
    def read_data_set(cls, path, batch_size, is_training):
        data_dict = np.load(path).item()
        dataset = tf.data.Dataset.from_tensor_slices(data_dict)
        if is_training:
            dataset = dataset.shuffle(100000).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)
        return dataset


def main():
    train_data = DataSet.read_data_set("data/train_data.npy", FLAGS.batch_size, True)
    test_data = DataSet.read_data_set("data/test_data.npy", FLAGS.test_neg+1, False)
    train(train_data, test_data, DataSet.user_size, DataSet.item_size)


if __name__ == '__main__':
    main()
