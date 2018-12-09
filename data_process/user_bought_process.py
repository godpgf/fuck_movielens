import pandas as pd
import numpy as np
import os
if not os.path.exists(os.path.join(os.getcwd(), "data")):
    os.chdir("..")


def read_user_bought(filename, sep="\t"):
    user_list = []
    bought_list = []
    ratings_list = []
    with open(filename, 'r') as f:
        f.readline()
        line = f.readline()
        while line:
            tmp = line[:-1].split(sep)
            if len(user_list) == 0 or int(tmp[0]) != user_list[-1]:
                user_list.append(int(tmp[0]))
                bought_list.append([int(tmp[1])])
                ratings_list.append([int(tmp[2])])
            else:
                bought_list[-1].append(int(tmp[1]))
                ratings_list[-1].append(int(tmp[2]))
            line = f.readline()

    item_popularity = dict()
    for i in range(len(user_list)):
        items = bought_list[i]
        sum_rating = 0
        for r in ratings_list[i]:
            sum_rating += r
        for index, item in enumerate(items):
            item_popularity.setdefault(item, 0.0)
            item_popularity[item] += ratings_list[i][index] / float(r)
    item_popularity = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)
    item = []
    popularity = []
    for data in item_popularity:
        item.append(data[0])
        popularity.append(data[1])

    df_bought = pd.DataFrame({"user_id": np.array(user_list), "items": bought_list, "ratings": ratings_list})
    df_popularity = pd.DataFrame({"item_id": item, "popularity": popularity})
    return df_bought, df_popularity


if __name__ == '__main__':
    df_bought, df_item_popularity = read_user_bought("data/train_ratings.csv", ",")
    df_bought.to_csv("data/user_bought_train.csv", index=False)
    df_item_popularity.to_csv("data/item_popularity_train.csv", index=False)
    df_bought, df_item_popularity = read_user_bought("data/test_ratings.csv", ",")
    df_bought.to_csv("data/user_bought_test.csv", index=False)


