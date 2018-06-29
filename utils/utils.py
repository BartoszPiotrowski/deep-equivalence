import os, pickle, re
from time import strftime
from random import shuffle
from datetime import datetime


def save_obj(obj, filename):
    print("Saving given object to file {}".format(filename))
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(filename):
    print("Loading object {}".format(filename))
    with open(filename, 'rb') as f:
        return pickle.load(f)

def readlines(filename):
    with open(filename, encoding ='utf-8') as f:
        return f.read().splitlines()

def writelines(list_of_lines, filename):
    with open(filename, encoding ='utf-8', mode='wt') as f:
        f.write('\n'.join(list_of_lines))

def list_to_line(l):
    return ', '.join([str(i) for i in l])

def mkdir_if_not_exists(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def printline(message, logfile='', verbose=True, time=True):
    if verbose:
        print(message)
    if logfile:
        with open(logfile, 'a') as f:
            if time:
                message = "[{}] {}".format(strftime("%Y-%m-%d %H:%M:%S"), message)
            print(message, file=f)

def shuffled(l):
    shuffle(l)
    return l

def partition(lst, n):
    if n > len(lst):
        n = len(lst)
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]

def make_path(directory=None, file=None, params=None):
    assert params
    assert directory or file
    params_str = '{}--{}'.format(
        datetime.now().strftime('%Y-%m-%d_%H%M%S'),
        ','.join(
            ('{}={}'.format(
                re.sub("(.)[^_]*_?", r"\1", key), value)
                for key, value in params.items())))
    if directory:
        return os.path.join(directory, params_str)
    if file:
        return file + params_str
