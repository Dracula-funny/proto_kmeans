import torch
import numpy as np


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch   #100
        self.n_cls = n_cls       #20
        self.n_per = n_per       #5+15

        label = np.array(label)
        self.m_ind = []                         #m_ind里面装的是从标签0开始所有的下标   mind是二维的   第一次加进去的是标签维0的所有元素下标
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)   # 返回label=i的所有下标  第一次共600个 0-599
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch


    #每次挑20类图片，每类20张，一共400张返回
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]    #生成0-100的所有数字打乱之后拿前二十个放进classes    20way   数据集有100类
            #第一次的classes  tensor([11, 10, 36, 25,  9,  4, 31, 47, 48, 34, 35, 30,  1, 57,  5, 18, 41, 29,13, 56])
            for c in classes:
                #第二次循环 c=10   l:  tensor([6000~6599])
                l = self.m_ind[c]                                      #将标签为c的数据都放到l里边
                #pos:  tensor([145,499,239,.....])    len(l)=600   pos里面只有20个
                pos = torch.randperm(len(l))[:self.n_per]            #600里面挑出20个，这20个的位置放进pos
                batch.append(l[pos])                                   #一次取了20张图片放进去batch
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

