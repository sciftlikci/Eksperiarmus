from collections import OrderedDict
from numpy.random import choice
from numpy.random import seed
import numpy as np
import textdistance
import pandas as pd
from scipy.stats import *


# Rank reversal probability function
def getProbability(array, position):
    return 1 - sum([1 / np.size(array) for i in array if array[position] == i])


# longest common substring similarity
def text_similarity(text1, text2):
    return textdistance.lcsstr.normalized_similarity(text1, text2)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    rng = se * t.ppf((1 + confidence) / 2., n - 1)
    return str([round(100 * (m - rng), 2), round(100 * (m + rng), 2)])


# Using map() and lambda
def listOfTuples(l1, l2):
    return list(map(lambda x, y: (x, y), l1, l2))


def init_medoids(X, k):
    seed(1)
    return list(choice(len(X), size=k, replace=False))


def downstream(matrix_V, a):
    data = OrderedDict()
    neg_sol = np.amin(matrix_V, axis=0)
    pos_sol = np.amax(matrix_V, axis=0)
    converter = {0: 1, 1: 0}
    mask1 = np.multiply(neg_sol, a)  # [min1, min2, min3, min4] * [1, 0, 1, 0]
    mask2 = np.multiply(pos_sol,
                        [converter[element] for element in a])  # [max1, max2, max3, max4] * [0, 1, 0, 1]
    data['+ve'] = np.where(mask2 == 0, mask1, mask2)  # positive ideal

    mask1 = np.multiply(neg_sol, [converter[element] for element in a])
    mask2 = np.multiply(pos_sol, a)
    data['-ve'] = np.where(mask2 == 0, mask1, mask2)  # negative ideal

    return data


# Calculation of Separation Measure
def midstream(matrix_V, L):
    data = OrderedDict()
    data['+ve'] = np.sqrt(np.sum(np.square(np.subtract(matrix_V, L['+ve'])), axis=1))
    data['-ve'] = np.sqrt(np.sum(np.square(np.subtract(matrix_V, L['-ve'])), axis=1))

    return data


# Calculating relative closeness to the Ideal Solution
def upstream(S):
    return np.divide(S['-ve'], np.add(S['+ve'], S['-ve']))


# datetime transform
def datetime_transform(df):
    return (pd.to_datetime(df['RAPOR_TARIHI']) - pd.to_datetime(df['EKSPER_ATAMA_TARIHI'])).dt.days + 1


# eliminate outliers
def eliminate_outliers(df, outlier=False):
    upper_limit_cost = 200000
    lower_limit_cost = 50
    upper_limit_time = 60

    df = df[(df["TOTAL_FILE_COST"] <= upper_limit_cost) & (df["TOTAL_FILE_COST"] >= lower_limit_cost)]
    df["FILE CLOSURE TIME"] = datetime_transform(df)
    df = df[df["FILE CLOSURE TIME"] <= upper_limit_time]
    df.drop("FILE CLOSURE TIME", axis=1, inplace=True)

    return df


def desirability_mean_downstream(x, t_c, l_c, stb):
    if stb:
        return 1 if x < t_c else ((x - l_c) / (t_c - l_c)) ** 1 if t_c <= x <= l_c else 0
    else:
        return 0 if x < l_c else ((x - l_c) / (t_c - l_c)) ** 1 if l_c <= x <= t_c else 1


def desirability_exponent_downstream(mean, std):
    return (mean ** 0.8) * (std ** 0.2)


def desirability_ltb_downstream(x, t_c, l_c):
    return 0 if x < l_c else ((x - l_c) / (t_c - l_c)) ** 1 if l_c <= x <= t_c else 1


def desirability_stb_downstream(x, t_c, u_c):
    return 1 if x < t_c else ((x - u_c) / (t_c - u_c)) ** 1 if t_c <= x <= u_c else 0
