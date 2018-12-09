import time
import numpy as np
import tensorflow as tf
import pandas as pd

from wide_deep_model import WideDeepModel
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
tf.app.flags.DEFINE_integer('epochs', 50, 'the number of epochs.')
tf.app.flags.DEFINE_integer('topK', 10, 'topk for evaluation.')
tf.app.flags.DEFINE_string('optim', 'Adam', 'the optimization method.')
tf.app.flags.DEFINE_string('initializer', 'Xavier', 'the initializer method.')
tf.app.flags.DEFINE_string('loss_func', 'cross_entropy', 'the loss function.')
tf.app.flags.DEFINE_string('activation', 'ReLU', 'the activation function.')
tf.app.flags.DEFINE_string('model_dir', 'model/', 'the dir for saving model.')
tf.app.flags.DEFINE_float('regularizer', 0.0, 'the regularizer rate.')
tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate.')
tf.app.flags.DEFINE_float('dropout', 0.0, 'dropout rate.')


def train(train_data, train_len, test_data, user_size,item_size):
    with tf.Session() as sess:
        iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                   train_data.output_shapes)

        print(train_data.output_types, train_data.output_shapes)
        model = WideDeepModel(FLAGS.embedding_size, user_size, item_size, [64, 64, 16], FLAGS.lr,
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

        # 锁定图，不让它在循环中更改
        # sess.graph.finalize()

        sess.run(model.iterator.make_initializer(train_data))
        for epoch in range(FLAGS.epochs):

            model.is_training = True
            model.get_data()
            start_time = time.time()

            for count in range(train_len // FLAGS.batch_size):
                model.step(sess, count)
                count += 1
            print("Epoch %d training " % epoch + "Took: " + time.strftime("%H: %M: %S",
                                                                              time.gmtime(time.time() - start_time)))

        sess.run(model.iterator.make_initializer(test_data))
        model.is_training = False
        model.get_data()
        start_time = time.time()
        HR,MRR,NDCG = [],[],[]
        # prediction, label = model.step(sess, None)
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
            print("HR is %.3f, MRR is %.3f, NDCG is %.3f" % (hr, mrr, ndcg))

        ################################## SAVE MODEL ################################
        checkpoint_path = os.path.join(FLAGS.model_dir, "NCF.ckpt")
        model.saver.save(sess, checkpoint_path)


class DataSet(object):
    user_size = 6040
    item_size = 3706

    @classmethod
    def read_data_set(cls, path, batch_size, is_training):
        data_dict = np.load(path).item()
        # 得到用户数据
        df_user = pd.read_csv("data/ml-1m/users.dat", sep="::", header=None, names=["user_id", "gender", "age", "occupation", "zip-code"],
                         usecols=[0, 1, 2, 3, 4], dtype={0: np.int32, 2: np.int32, 3: np.int32}, engine='python')
        df_user = pd.DataFrame({"user_id": data_dict['user']}).merge(df_user, how="left", on="user_id")
        df_user["gender"] = df_user["gender"].apply(lambda gender: 1 if gender == 'M' else 0)
        # age_dic = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
        age_dic = {value: np.int32(key) for key, value in enumerate(set(df_user["age"].values.tolist()))}
        df_user["age"] = df_user["age"].apply(lambda age: age_dic[age])

        # 得到电影数据
        df_item = pd.read_csv("data/ml-1m/movies.dat", sep="::", header=None, names=["item_id", "titles", "genres"], engine='python')
        def _map_fn(entry):
            # entry = entry.replace("Children's", "Children")  # naming difference.
            movie_genres = entry.split("|")
            # output = np.zeros((len(GENRES),), dtype=np.int64)
            output = [0.0] * len(GENRES)
            for i, genre in enumerate(GENRES):
                if genre in movie_genres:
                    output[i] = 1.0
            return output
        df_item["genres"] = df_item["genres"].apply(_map_fn)
        # print(pd.DataFrame({"item_id": data_dict['item']})[["item_id"]])
        df_item = pd.DataFrame({"item_id": data_dict['item']}).merge(df_item, how="left", on="item_id")

        data_dict = dict([('user', np.array(data_dict['user'], dtype=np.int32)), ('age', df_user['age'].values.astype(np.int32)), ('gender', df_user['gender'].values.astype(np.int32)), ('occupation', df_user['occupation'].values.astype(np.int32)),
                          ('item', np.array(data_dict['item'], dtype=np.int32)), ('genres', np.array(df_item['genres'].values.tolist(), dtype=np.float32)),
                          ('label', np.array(data_dict['label'], dtype=np.float32))])
        dataset = tf.data.Dataset.from_tensor_slices(data_dict)
        if is_training:
            dataset = dataset.shuffle(100000).batch(batch_size).repeat()
        else:
            dataset = dataset.batch(batch_size)
        return dataset, len(data_dict['label'])


def main():
    train_data, train_len = DataSet.read_data_set("data/train_data.npy", FLAGS.batch_size, True)
    test_data, _ = DataSet.read_data_set("data/test_data.npy", FLAGS.test_neg+1, False)
    train(train_data, train_len, test_data, DataSet.user_size, DataSet.item_size)


if __name__ == '__main__':
    main()
