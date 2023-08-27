import argparse
import os.path as osp
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from convnet import Convnet
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric,getprotoconfi,getknnconfi
import random
import pandas as pd
import numpy
from datetime import datetime

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

#创建train_acc.csv和var_acc.csv文件，记录loss和accuracy
df = pd.DataFrame(columns=['time','step','train Loss','training accuracy','knn_acc','knn_loss','kvl_acc'])#列名
df.to_csv("train_acc.csv",index=False) #路径可以根据需要更改


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=5)        #1
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=20)   #30
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='./save/proto-5')      #./save/proto-1
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    ensure_path(args.save_path)

    trainset = MiniImageNet('train')  #所有训练数据  38400个
    train_sampler = CategoriesSampler(trainset.label, 100,
                                      args.train_way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)

    valset = MiniImageNet('val')
    val_sampler = CategoriesSampler(valset.label, 400,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=True)

    model = Convnet().cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    def save_model(name):
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))
    
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    # for i, batch in enumerate(train_loader, 1):
    #     print(i)
    #     print(batch)
    #     print(np.array(batch).shape())
    #     print('..........................')

    for epoch in range(1, args.max_epoch + 1):
        lr_scheduler.step()

        model.train()

        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader, 1):

            data, _ = [_.cuda() for _ in batch]         #利用for循环吧batch中的内容装入data  (400,3,84,84)
                                                        #每个batch里面有400张图片,分别存放了图片和相应的标签
            p = args.shot * args.train_way
            data_shot, data_query = data[:p], data[p:]   #datashot:(100,3,84,84) dataquery:(300,3,84,84)

            proto = model(data_shot)                    #data_shot有100张，20类
                                                        #这里的proto是  (5,20,1600),在下面按照第0维求了个平均值所以下面是(20,1600)


            #接下来的任务是把这个三维的数据输入到KNN看一下结果

            #1.处理KNN需要的训练数据和标签
            knn_train = torch.reshape(proto,(-1,1600))          #训练数据
            knn_train = knn_train.cpu().detach().numpy()
            knn_label = torch.arange(args.train_way).repeat(args.shot)
            knn_label = knn_label.type(torch.cuda.LongTensor)    #标签
            knn_label = knn_label.cpu().detach().numpy()
            knn_test_label = torch.arange(args.train_way).repeat(args.query)
            knn_test_label = knn_test_label.type(torch.cuda.LongTensor)  # 标签
            knn_test_label = knn_test_label.cpu().detach().numpy()

            knn_test = torch.reshape(model(data_query),(-1,1600))
            knn_test = knn_test.cpu().detach().numpy()


            knn_loss = 0

            nearest_neighbors = NearestNeighbors(n_neighbors=3).fit(knn_train)

            distances, indices = nearest_neighbors.kneighbors(knn_test)

            # print("indices",indices)
            # print("indices.size", indices.shape)    #indices.size (300, 3)
            l=0
            for j in indices:

                k1 = 0
                for k in j:
                    if(knn_label[k] != knn_test_label[l]):
                           k1+=1

                knn_loss+=math.exp(k1/3)
                l+=1

            knn_loss /= 300

            # print(knn_label.shape)  #(100,)
            # print(knn_train.shape)   #(100, 1600)
            # print(knn_test_label.shape)   #(300,)
            # print(knn_test.shape)    #(300, 1600)


            proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)   #proto:(5,20,1600)

            label = torch.arange(args.train_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)+knn_loss
            acc = count_acc(logits, label)
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                  .format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            proto = None; logits = None; loss = None

        tl = tl.item()
        ta = ta.item()

        model.eval()

        vl = Averager()
        va = Averager()
        ka = Averager()
        kl = Averager()
        kvl = Averager()
        for i, batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)

            # 接下来的任务是把这个三维的数据输入到KNN看一下结果

            # 1.处理KNN需要的训练数据和标签
            knn_train = torch.reshape(proto, (-1, 1600))  # 训练数据
            knn_train = knn_train.cpu().detach().numpy()
            knn_label = torch.arange(args.test_way).repeat(args.shot)
            knn_label = knn_label.type(torch.cuda.LongTensor)  # 标签
            knn_label = knn_label.cpu().detach().numpy()
            knn_test_label = torch.arange(args.test_way).repeat(args.query)
            knn_test_label = knn_test_label.type(torch.cuda.LongTensor)  # 标签
            knn_test_label = knn_test_label.cpu().detach().numpy()

            #print(knn_test_label)               #[0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 34]

            # 创建分类器
            clf = KNeighborsClassifier(n_neighbors=3)

            # 训练数据
            clf.fit(knn_train, knn_label)  # fit的参数一个是训练数据，一个是标签

            # 处理测试数据


            knn_test = torch.reshape(model(data_query), (-1, 1600))
            knn_test = knn_test.cpu().detach().numpy()
            test_predictions = clf.predict(knn_test)

            #print(test_predictions)            #[0 2 1 2 2 0 0 2 2 3 2 0 2 0 2 0 0 2 2 2 2 2 2 2 3 2 2 1 2 0 0 2 0 1 2 2 12 2 2 0 2 2 2 2 0 2 1 1 0 2 1 1 2 3 2 0 0 2 2 3 2 0 2 0 0 2 0 2 0 2 0 2 0 0]
            # print(knn_test_label)
            # print(test_predictions)
            #print('Accuracy:', accuracy_score(knn_test_label, test_predictions))


            nearest_neighbors = NearestNeighbors(n_neighbors=3).fit(knn_train)

            distances, indices = nearest_neighbors.kneighbors(knn_test)
            # print(indices)

            l = 0
            knn_loss = 0
            for j in indices:
                # print(j)
                # print(knn_label[i])
                # print(knn_test_label[index])
                k1 = 0
                for k in j:
                    if (knn_label[k] != knn_test_label[l]):
                        k1 += 1
                # print(k1)
                knn_loss += math.exp(k1 / 3)
                l += 1
            # print(knn_loss)
            knn_loss /= 75

            proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

            label = torch.arange(args.test_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            # print("model(data_query)",model(data_query))
            # print("model(data_query)size", model(data_query).shape)    #torch.Size([75, 1600])

            # print("proto",proto)
            # print("protosize", proto.shape)  #protosize torch.Size([5, 1600])
            # tensor([[1.5523, 1.3623, 1.3501, ..., 0.4501, 0.3881, 1.5449],
            #         [1.5708, 1.2429, 1.3225, ..., 0.3061, 0.3505, 1.4340],
            #         [1.6898, 1.3082, 1.3249, ..., 0.2660, 0.2763, 1.2787],
            #         [1.5381, 1.3091, 1.2878, ..., 0.3163, 0.4382, 1.3940],
            #         [1.5167, 1.3537, 1.3463, ..., 0.3668, 0.2980, 1.3681]],
            #        device='cuda:0', grad_fn= < MeanBackward1 >)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)

            logits2 = torch.exp(logits)       #对logits中的数据求自然对数

            proto_confi = getprotoconfi(logits2,label)     #获得原型网络的置信度
            knn_confi = getknnconfi(knn_train,knn_label,knn_test,knn_test_label)       #获得knn的置信度

            pred = torch.argmax(logits, dim=1)  # 找出每行最大的索引值,也就是原型的预测结果
            pred = pred.tolist()
            test_predictions = test_predictions.tolist()


            # print("proto_confi",proto_confi)
            # print("knn_confi",knn_confi)
            # print("proto_pred",pred)
            # print("knn_pred",test_predictions)


            new_pred = []
            for m1,m2 in enumerate(proto_confi):
                if(proto_confi[m1]>knn_confi[m1]):
                    new_pred.append(pred[m1])
                else:
                    new_pred.append(test_predictions[m1])




            #print("new_pred",new_pred)
            # print(len(new_pred))

            new_acc = (new_pred == knn_test_label).mean().item()

            #print(new_acc)


            # print("label",label)
            # print("labelSize", label.shape)     #torch.Size([75])
            # print("logits",logits)
            # print("logitssize", logits.shape)    #torch.Size([75, 5])

            # logitstensor([[-10.6292, -10.8905, -11.2078, -10.9887, -8.9474],
            #         [-15.2012, -13.1727, -14.2442, -15.9999, -15.5249],
            #         [-10.3820, -7.8934, -7.7382, -8.5529, -11.2931],
            #         [-12.2339, -12.2158, -10.9883, -11.5106, -11.7058],
            #         [-15.5472, -17.2717, -16.2328, -16.6806, -14.8121],
            acc = count_acc(logits, label)
            knn_acc = accuracy_score(knn_test_label, test_predictions)

            vl.add(loss.item())
            va.add(acc)
            ka.add(knn_acc)
            kl.add(knn_loss)
            kvl.add(new_acc)
            proto = None; logits = None; loss = None; knn_acc = None;

        vl = vl.item()
        va = va.item()
        ka = ka.item()
        kl = kl.item()
        kvl = kvl.item()
        print('epoch {}, val, loss={:.4f} acc={:.4f}, knn_acc={:.4f},knn_loss={:.4f}, kvl_acc={:.4f}'.format(epoch, vl, va, ka, kl,kvl))

        time = "%s" % datetime.now()  # 获取当前时间
        step = "Step[%d]" % epoch


        # 将数据保存在一维列表
        list = [time, step, vl, va,ka,kl,kvl]
        # 由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
        data = pd.DataFrame([list])
        data.to_csv('train_acc.csv', mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了



        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))

        save_model('epoch-last')

        if epoch % args.save_epoch == 0:
            save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))

