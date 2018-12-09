import pandas as pd


def read_data_train_set(path):
    df = pd.read_csv(path)
    df["items"] = df["items"].apply(lambda entry: [int(d) for d in entry[1:-1].split(',')])
    df["ratings"] = df["ratings"].apply(lambda entry: [int(d) for d in entry[1:-1].split(',')])
    user_id = df["user_id"].values
    items = df["items"].values
    ratings = df["ratings"].values
    data_dic = {}
    for id, user in enumerate(user_id):
        data_dic[user] = {}
        for index, item in enumerate(items[id]):
            data_dic[user][item] = ratings[id][index]
    return data_dic


def read_data_test_set(path):
    df = pd.read_csv(path)
    df["items"] = df["items"].apply(lambda entry: [int(d) for d in entry[1:-1].split(',')])
    df["ratings"] = df["ratings"].apply(lambda entry: [int(d) for d in entry[1:-1].split(',')])
    user_id = df["user_id"].values
    items = df["items"].values
    ratings = df["ratings"].values
    data_dic = {}
    for id, user in enumerate(user_id):
        data_dic[user] = []
        for index, item in enumerate(items[id]):
            data_dic[user].append((item, ratings[id][index]))
    return data_dic


def df_2_dic(df):
    dic = {}
    for row in df[["user_id", "item_id", "rating"]].values:
        user, item, record = int(row[0]), int(row[1]), row[2]
        if user in dic:
            dic[user][item] = record
        else:
            dic[user] = {item: record}
    # for index, row in df.iterrows():
    #     user, item, record = int(row["user_id"]), int(row["item_id"]), row["rating"]
    #     dic.setdefault(user, {})
    #     dic[user][item] = record
    return dic


def get_all_items(dic):
    all_items = set()
    for user, items in dic.items():
        for i in items.keys():
            all_items.add(i)
    return all_items

