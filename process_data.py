import numpy as np
import pickle5 as pickle
import sys


def load_data(year, id):
    # 根据 id 选择文件
    if id == 50:
        file_path = "data/TS_Data50.pkl"
    elif id == 300:
        file_path = "data/TS_Data300.pkl"
    elif id == 500:
        file_path = "data/TSCL_500data_2DayRep.pkl"
    else:
        raise ValueError(f"Invalid id: {id}")

    # 加载数据
    word = pickle.load(open(file_path, 'rb'), encoding='utf-8')
    word = word.dropna()

    raw_data = word['FCLOSE']
    feature = word[['OPEN', 'CLOSE', 'HIGH', 'LOW', 'VOL', 'FOPEN', 'FCLOSE', 'FHIGH', 'FLOW', 'FVOL']].values.astype(float)
    word, raw_time = word['FCLOSE'].to_numpy(), word['DATE'].to_numpy()

    if year == 2020:
        start_time, end_time = 256600, 315000
    elif year == 2021:
        start_time, end_time = 315000, 373000
    elif year == 2022:
        start_time, end_time = 373000, 400000
    else:
        raise ValueError(f"Invalid year: {year}")

    data, time_data = word[start_time:end_time], raw_time[start_time:end_time]
    feature = feature[start_time:end_time]

    true_label = list()
    threshold, threshold2, window_size = 0.0004, 0.0002, 10
    for i in range(len(data)):
        total = sum(data[i:i + window_size]) / window_size
        ratio = float(data[i]) / float(total) - 1
        if ratio > threshold:
            temp = 2
        elif threshold >= ratio > threshold2:
            temp = 1
        elif threshold2 >= ratio > -threshold2:
            temp = 0
        elif -threshold2 >= ratio > -threshold:
            temp = -1
        elif ratio <= -threshold:
            temp = -2
        true_label.append(temp)

    true_label = np.expand_dims(true_label, axis=1)
    combined_array = np.concatenate((feature, true_label), axis=1)
    print(combined_array.shape)
    save_path = f'./newdata/{id}_{year}.npy'
    np.save(save_path, combined_array)


if __name__ == "__main__":
    # year, id = 2022, 50
    year = int(sys.argv[1])
    id = int(sys.argv[2])
    load_data(year, id)