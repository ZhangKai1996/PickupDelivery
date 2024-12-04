import math
import pandas as pd


def run1(args, gamma=0.99, l=100):
    w, y, *_ = args
    delta = math.pow(gamma, l) * (1 + gamma)
    delta *= (1-gamma) * w - y
    return round(delta, 3)


def g(gamma, y, num):
    sum_v = 0.0
    for i in range(num):
        sum_v += math.pow(gamma, i) * y
    return sum_v


def run2(args, gamma=0.995, T=100):
    w, y, *_ = args
    delta = math.pow(gamma, T) * (w - y)
    return round(delta, 3)

def main():
    row = 30
    column = 10

    data, index = [], []
    for i in range(column):
        value = (100 * (i + 1), -1 * (i + 1))
        vector = {}
        for j in range(row):
            vector[int(j + 1)] = run2(value, T=(j + 1) * 30)
            # vector[int(j + 1)] = run1(value, l=(j + 1) * 30)
        index.append(int(i + 1))
        data.append(vector)
        print(i+1, value, vector)

    df = pd.DataFrame(data, index=index)
    df.to_csv('result.csv')
    print(df)


if __name__ == '__main__':
    main()