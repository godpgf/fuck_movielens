import pandas as pd
import numpy as np
import os
import sys
if not os.path.exists(os.path.join(os.getcwd(), "data")):
    os.chdir("..")
# 分测试集和训练集


def read_leave_one_out_ratings_file(filename, sep="\t"):
    train_user_id = []
    test_user_id = []
    train_item_id = []
    test_item_id = []
    train_rating = []
    test_rating = []
    train_timestamp = []
    test_timestamp = []

    def write_train_and_test(cur_user_ratings):
        if len(cur_user_ratings) > 1:
            for i in range(len(cur_user_ratings) - 1):
                train_user_id.append(cur_user_ratings[i][0])
                train_item_id.append(cur_user_ratings[i][1])
                train_rating.append(cur_user_ratings[i][2])
                train_timestamp.append(cur_user_ratings[i][3])
            test_user_id.append(cur_user_ratings[-1][0])
            test_item_id.append(cur_user_ratings[-1][1])
            test_rating.append(cur_user_ratings[-1][2])
            test_timestamp.append(cur_user_ratings[-1][3])

    with open(filename, 'r') as f:
        line = f.readline()
        cur_user_ratings = []
        while line:
            tmp = line.replace('\r', '').replace('\n', '').split(sep)
            if len(cur_user_ratings) == 0 or int(tmp[0]) == cur_user_ratings[0][0]:
                cur_user_ratings.append([int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3])])
            else:
                write_train_and_test(cur_user_ratings)
                cur_user_ratings = [[int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3])]]
            line = f.readline()
        write_train_and_test(cur_user_ratings)

    df_train = pd.DataFrame({"user_id": np.array(train_user_id), "item_id": np.array(train_item_id), "rating": np.array(train_rating), "timestamp": np.array(train_timestamp)})
    df_test = pd.DataFrame(
        {"user_id": np.array(test_user_id), "item_id": np.array(test_item_id), "rating": np.array(test_rating),
         "timestamp": np.array(test_timestamp)})
    return df_train, df_test


def read_time_split_ratings_file(filename, sep="\t", ratio=0.016):
    data_list = []
    with open(filename, 'r') as f:
        line = f.readline()
        cur_user_ratings = []
        while line:
            tmp = line[:-1].split(sep)
            data_list.append((int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3])))
            line = f.readline()
    data_list = np.array(sorted(data_list, key=lambda c: c[3]), dtype=[('user_id', int), ('item_id', int), ('rating', int), ('timestamp', int)])
    train_size = int(len(data_list) * (1 - ratio))
    # train_data = sorted(data_list[:train_size], cmp=lambda a,b: -1 if a[0] < b[0] else (1 if a[0] > b[0] else a[3] - b[3]))
    # test_data = sorted(data_list[train_size:], cmp=lambda a,b: -1 if a[0] < b[0] else (1 if a[0] > b[0] else a[3] - b[3]))
    train_data = np.sort(data_list[:train_size], order=['user_id', 'timestamp'])
    test_data = np.sort(data_list[train_size:], order=['user_id', 'timestamp'])
    df_train = pd.DataFrame({"user_id": train_data['user_id'], "item_id": train_data['item_id'], "rating": train_data['rating'], "timestamp": train_data['timestamp']})
    df_test = pd.DataFrame({"user_id": test_data['user_id'], "item_id": test_data['item_id'], "rating": test_data['rating'], "timestamp": test_data['timestamp']})
    return df_train, df_test


if __name__ == '__main__':
    ratio = None
    if len(sys.argv) > 1:
        ratio = float(sys.argv[1])
    if ratio is None:
        df_train, df_test = read_leave_one_out_ratings_file("data/ml-1m/ratings.dat", "::")
    else:
        df_train, df_test = read_time_split_ratings_file("data/ml-1m/ratings.dat", "::", ratio)
    df_train.to_csv("data/train_ratings.csv", index=False)
    df_test.to_csv("data/test_ratings.csv", index=False)
