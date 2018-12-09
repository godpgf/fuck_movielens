import numpy as np
import pandas as pd


def rating_clean(filename, sep="::"):
    # 重新给item分配id
    cur_item_id = 0
    item_re_index = {}

    data_list = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            tmp = line.replace('\r', '').replace('\n', '').split(sep)
            data_list.append((int(tmp[0]) - 1, int(tmp[1]), int(tmp[2]), int(tmp[3])))

            if int(tmp[1]) not in item_re_index:
                item_re_index[int(tmp[1])] = cur_item_id
                cur_item_id += 1

            line = f.readline()
    data_list = np.array(sorted(data_list, key=lambda c: c[3]), dtype=[('user_id', int), ('item_id', int), ('rating', int), ('timestamp', int)])
    data_list = np.sort(data_list, order=['user_id', 'timestamp'])
    with open(filename, 'w') as f:
        for data in data_list:
            f.write(sep.join([str(data[0]), str(item_re_index[data[1]]), str(data[2]), str(data[3])]) + '\n')
    return item_re_index


def movie_clean(filename, item_re_index, sep="::"):
    df = pd.read_csv(filename, sep=sep, header=None,names=["id", "title", "genres"],
                            usecols=[0,1,2],dtype={0:np.int32},engine='python')
    df = df[df['id'].isin(item_re_index)]
    df["id"] = df["id"].apply(lambda entry: item_re_index[entry])
    df = df.sort_values(by=["id"], ascending=[True])

    # df.to_csv(filename, sep=sep, header=None, index=False)
    with open(filename, 'w') as f:
        for _, row in df.iterrows():
            f.write("%d::%s::%s\n"%(row['id'], row['title'], row['genres']))


def user_clean(filename, sep="::"):
    df = pd.read_csv(filename, sep=sep, header=None,names=["id", "gender", "age", "occupation", "zip-code"],
                            usecols=[0,1,2,3,4],dtype={0:np.int32},engine='python')
    df["id"] = df["id"] - 1
    # df.to_csv(filename, sep=sep, header=None, index=False)
    with open(filename, 'w') as f:
        for _, row in df.iterrows():
            f.write("%d::%s::%d::%s::%s\n" % (row['id'], row['gender'], row['age'], row['occupation'], str(row['zip-code'])))


if __name__ == '__main__':
    user_clean("users.dat")
    item_re_index = rating_clean("ratings.dat")
    movie_clean("movies.dat", item_re_index)

