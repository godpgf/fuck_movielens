import pandas as pd
import numpy as np
import os
import pickle
import random

if not os.path.exists(os.path.join(os.getcwd(), "data")):
    os.chdir("..")


def add_negative(features, user_negative, labels, numbers, is_training):
    feature_user, feature_item, labels_add, feature_dict = [], [], [], {}

    for i in range(len(features)):
        user = features['user_id'][i]
        item = features['item_id'][i]
        label = labels[i]

        feature_user.append(int(user))
        feature_item.append(int(item))
        labels_add.append(int(label))

        if is_training:
            negative = user_negative[user][:]
            if numbers < len(negative):
                # neg_samples = negative[:numbers]
                neg_samples = np.random.choice(negative, size=numbers, replace=False).tolist()
            else:
                # neg_samples = negative
                neg_samples = np.random.choice(negative, size=numbers).tolist()
        else:
            bad_item = None
            for i in range(len(user_negative[user])-1, -1, -1):
                bad_item = user_negative[user][i]
                if bad_item != item:
                    break
            negative = set(user_negative[user][:])
            if item in negative:
                negative.remove(item)
            negative = list(negative)
            if len(negative) < numbers:
                negative.extand([bad_item] * (numbers - len(negative)))
            neg_samples = np.array(negative)

        if is_training:
            for k in neg_samples:
                feature_user.append(int(user))
                feature_item.append(int(k))
                labels_add.append(int(0))

        else:
            for k in neg_samples:
                feature_user.append(int(user))
                feature_item.append(int(k))
                labels_add.append(int(k))

    feature_dict['user'] = feature_user
    feature_dict['item'] = feature_item
    feature_dict['label'] = labels_add
    return feature_dict


def read_user_bought(path):
    df = pd.read_csv(path)
    df["items"] = df["items"].apply(lambda entry: [int(d) for d in entry[1:-1].split(',')])
    user_id = df["user_id"].values
    items = df["items"].values
    data_dic = {}
    for id, user in enumerate(user_id):
        data_dic[user] = []
        for index, item in enumerate(items[id]):
            data_dic[user].append(int(item))
    return data_dic


NEG_NUM = 4
TEST_NEG_NUM = 3705


if __name__ == '__main__':
    user_negative = read_user_bought("data/user_negative_train.csv")
    data = pd.read_csv("data/train_ratings.csv")
    labels = np.ones(len(data), dtype=np.int32)
    train_dic = add_negative(data, user_negative, labels, NEG_NUM, True)
    np.save("data/train_data", train_dic)

    data = pd.read_csv("data/test_ratings.csv")
    labels = data['item_id'].tolist()
    test_dic = add_negative(data, user_negative, labels, TEST_NEG_NUM, True)
    np.save("Data/test_data", test_dic)
