import pandas as pd
import numpy as np
import math
import readers
from pprint import pprint

# Constant seed for replicating training results
np.random.seed(42)

import os
if not os.path.exists(os.path.join(os.getcwd(), "data")):
    os.chdir("..")


dic_train = readers.read_data_train_set("data/user_bought_train.csv")
dic_test = readers.read_data_test_set("data/user_bought_test.csv")

all_items = set(pd.read_csv("data/test_ratings.csv")['item_id'].unique())
print("完成所有物品统计")
item_popularity_dic = {int(row["item_id"]):float(row["popularity"]) for _, row in pd.read_csv("data/item_popularity_train.csv").iterrows()}
item_popularity_list = [(int(row["item_id"]), float(row["popularity"])) for _, row in pd.read_csv("data/item_popularity_train.csv").iterrows()]
print("完成物品流行度统计")


def copy_train_dict(train):
    # 复制一遍，因为评估过程中会动态修改
    train_copy = {}
    for key, value in train.items():
        train_copy[key] = {}
        for sub_key, sub_value in value.items():
            train_copy[key][sub_key] = sub_value
    return train_copy


def eval_test(train, test, all_items, item_popularity_list, item_popularity_dic, W, K, N):
    train = copy_train_dict(train)
    hr = 0
    ndcg = 0

    # 记录预测到的物品在总物品中的比重
    recommend_items = set()

    # 计算新鲜度:测评的最简单方法是利用推荐结果的平均流行度，越不热门的物品越可能让用户觉得新颖。返回值越小，新颖度越大
    ret = 0  # 新颖度结果
    n = 0  # 推荐的总个数
    for user, items in test.items():
        # tu = test.get(user, {})
        for item in items:
            rank = recommender(user, train, item_popularity_list, W, K, N)
            train.setdefault(user, {})
            # 更新训练集
            train[user][item[0]] = item[1]
            index = -1
            for i in range(len(rank)):
                if rank[i][0] == item[0]:
                    index = i
                recommend_items.add(rank[i][0])
                if item[0] in item_popularity_dic:
                    ret += math.log(1 + item_popularity_dic[item[0]])
                n += 1
            if index >= 0:
                hr += 1
                ndcg += np.reciprocal(np.log2(index + 2))
    ret /= n * 1.0
    # HR，NDCG,覆盖率，流行度
    recommend_num = n / N
    return hr / recommend_num, ndcg / recommend_num, len(recommend_items) / (len(all_items) * 1.0), ret


# 物品相似度
def itemSimilarity(train):
    # 两个物品之间的初步相似度
    cor_items = dict()
    # 物品被用户购买的次数
    n_users = dict()
    step = 0
    for u, items in train.items():
        step += 1
        print("%d/%d"%(step,len(train)))
        for i in items:
            # 记录某个物品被选中的次数
            n_users[i] = n_users.get(i, 0) + 1
            for j in items:
                if i == j:
                    continue
                cor_items.setdefault(i, {})
                # 一个用户如果选过太多物品，这个人就不会太有个性
                cor_items[i][j] = cor_items[i].get(j, 0) + 1 / math.log(1 + len(items) * 1.0)

    W = dict()
    for i, related_items in cor_items.items():
        max_wi = 0.0
        for j, cij in related_items.items():
            W.setdefault(i, {})
            W[i][j] = cij / math.sqrt(n_users[i] * n_users[j])
            max_wi = max(W[i][j], max_wi)
        # 归一化
        for j, cij in related_items.items():
            W[i][j] /= max_wi
        W[i]["sort_items"] = sorted(W[i].items(), key=lambda c: c[1], reverse=True)
    return W


# 推荐
def recommender(user, train, item_popularity_list, W, K, N):
    """recommend to user N item according to K max similarity item
        给用户推荐K个物品，物品来源于与用户偏好物品的N个最相似的物品
    """
    rank = dict()
    if user in train:
        interacted_items = train[user]
    else:
        interacted_items = {}
    if len(interacted_items) > 0:
        for i, pi in interacted_items.items():
            # 找到和用户user已经打分的物品i最相似的k个物品，item_id和相似度是(j, wj)
            if i in W:
                for j, wj in W[i]["sort_items"][0:K]:
                    if j in interacted_items:
                        continue
                    # 得到用户对物品j的喜好预测
                    rank[j] = rank.get(j, 0) + pi * wj
    # 从所有物品中取出n个
    for item, pop in item_popularity_list:
        if len(rank) >= N:
            break
        if item not in rank:
            rank[item] = pop

    return sorted(rank.items(), key=lambda c: c[1], reverse=True)[0:N]


W = itemSimilarity(dic_train)
rank = recommender(344, dic_train, item_popularity_list, W, 5, 10)
print("测试给id为344的用户推荐10部电影：")
pprint(rank)
# 打开文件,清空内容
result = open('result_ibcf.data', 'w')
print(u'不同K值下推荐算法的各项指标(命中率、折扣累积增益、覆盖率、流行度)\n')

print('K\t\tHR\t\tNDCG\t\tCoverage\tPopularity')

for k in [5, 10, 20, 40, 80, 160]:
    hr, ndcg, cov, pop = eval_test(dic_train, dic_test, all_items, item_popularity_list, item_popularity_dic, W, k, 10)
    print("%3d\t\t%.2f%%\t\t%.2f%%\t\t%.2f%%\t\t%.6f" % (k, hr * 100, ndcg * 100, cov * 100, pop))
    result.write(str(k) + ' ' + str('%2.2f' % (hr * 100)) + ' ' + str('%2.2f' % (ndcg * 100)) + ' ' + str(
        '%2.2f' % (cov * 100)) + ' ' + str('%2.6f' % pop) + '\n')
