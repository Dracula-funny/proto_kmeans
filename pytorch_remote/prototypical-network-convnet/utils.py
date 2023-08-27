import os
import shutil
import time
import pprint
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import train_test_split
import torch


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path):
    # if os.path.exists(path):
    #     if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
    #         shutil.rmtree(path)
    #         os.makedirs(path)
    # else:
    #     os.makedirs(path)
    shutil.rmtree(path)
    os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)    #找出每行最大的索引值
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


def dot_metric(a, b):
    return torch.mm(a, b.t())


def euclidean_metric(a, b):
    n = a.shape[0]                          #shape(0):读取第一维的长度     这里的意思是读取a的第一维的长度，a有多少张照片
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def getprotoconfi(logits2,label):
    x1=0
    confi = []
    logits2 = logits2.tolist()

    for i,j in enumerate(logits2):
        for k in j:
            x1+=k
        x2 = j[label[i]]
        x3 = x2/x1

        confi.append(x3)
    confisum = 0
    for l in confi:
        confisum+=l

    newconfi = [z/confisum for z in confi]
    return newconfi

def getknnconfi(knn_train,knn_train_lable,knn_test,knn_test_lable):

    confi = []
    nearest_neighbors = NearestNeighbors(n_neighbors=3).fit(knn_train)
    distances, indices = nearest_neighbors.kneighbors(knn_test)
    for x,y in enumerate(indices):
        x1=knn_test_lable[x]   #测试样本的真实标签
        sum = 0
        for y1 in y:
            if(knn_train_lable[y1]==x1):
                sum+=1
        confi.append(sum/3)
    confisum=0
    for l in confi:
        confisum+=l

    newconfi = [z/confisum for z in confi]
    return newconfi



class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def l2_loss(pred, label):
    return ((pred - label)**2).sum() / len(pred) / 2

