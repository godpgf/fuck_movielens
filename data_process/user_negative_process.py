import pandas as pd
import numpy as np
import os
import sys
if not os.path.exists(os.path.join(os.getcwd(), "data")):
    os.chdir("..")


def read_user_bought(path):
    df = pd.read_csv(path)
    df["items"] = df["items"].apply(lambda entry: [int(d) for d in entry[1:-1].split(',')])
    user_id = df["user_id"].values
    items = df["items"].values
    data_dic = {}
    for id, user in enumerate(user_id):
        data_dic[user] = []
        for index, item in enumerate(items[id]):
            data_dic[user].append(item)
    return data_dic


if __name__ == "__main__":
    user_bought = read_user_bought("data/user_bought_train.csv")
    item_set = set(pd.read_csv("data/train_ratings.csv")['item_id'].unique())
    item_popularity = {row['item_id']: row['popularity'] for _, row in
                       pd.read_csv("data/item_popularity_train.csv").iterrows()}
    user_id = []
    items = []
    for key in user_bought:
        cur_items = list(item_set - set(user_bought[key]))
        items.append(sorted(cur_items, key=lambda a:item_popularity[a], reverse=True))
        user_id.append(key)
    pd.DataFrame({"user_id": user_id, "items": items}).to_csv("data/user_negative_train.csv", index=False)


