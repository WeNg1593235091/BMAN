import datetime
starttime = datetime.datetime.now()
import random
import torch
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(precision=8)

from models.data_loader import JSONFileDataLoader
from models.framework import FewShotREFramework
from models.BMAN import BMAN as BMAN

seed = int(np.random.uniform(0,1)*10000000)  #random.uniform()返回一个随机的浮点数
torch.manual_seed(seed)  #在需要生成随机数的实验中，确保每次运行.py文件时，生成的随机数都是固定的，这样每次实验结果显示也就一致了。
torch.cuda.manual_seed(seed)  #torch.cuda.manual_seed(int.seed):为当前GPU设置随机种子
np.random.seed(seed) #seed()被设置了之后，np,random.random()可以按顺序产生一组固定的数组，如果使用相同的seed()值，则每次生成的随机数都相同，如果不设置这个值，那么每次生成的随机数不同。但是，只在调用的时候seed()一下并不能使生成的随机数相同，需要每次调用都seed()一下，表示种子相同，从而生成的随机数相同。
random.seed(seed)
print('seed: ', seed)
import argparse   #命令行选项、参数和子命令解析器。
'''
创建解析对象parser
为对象parser添加属性
将parser中的属性返回给argsS
'''
parser = argparse.ArgumentParser(description='Bidirectional Matching and Aggregation Network for Few-Shot Relation Extraction')  #创建解析器
parser.add_argument('--model_name', type=str, default='BMAN', help='Model name')   #添加参数
parser.add_argument('--N_for_train', type=int, default=20, help='Num of classes for each batch for training')  #每次训练种类的个数，default是设定参数的默认值
parser.add_argument('--N_for_test', type=int, default=5, help='Num of classes for each batch for test')
parser.add_argument('--K', type=int, default=1, help='Num of instances for each class in the support set') #每个种类中支持集的实例数
parser.add_argument('--Q', type=int, default=5, help='Num of instances for each class in the query set') #每个种类中查询集的实例数
parser.add_argument('--batch', type=int, default=1, help='batch size')   #一次训练中选取的样本数
parser.add_argument('--max_length', type=int, default=40, help='max length of sentence')  #句子的最大长度
parser.add_argument('--learning_rate', type=float, default=2e-1, help='initial learning rate')   #初始学习率

args = parser.parse_args()
print('setting:')
print(args)

print("{}-way(train)-{}-way(test)-{}-shot with batch {} Few-Shot Relation Classification".format(args.N_for_train, args.N_for_test, args.K, args.Q))
print("Model: {}".format(args.model_name))

max_length = args.max_length

train_data_loader = JSONFileDataLoader('./data/train.json', './data/glove.6B.50d.json', max_length=max_length, reprocess=False)
val_data_loader = JSONFileDataLoader('./data/val.json', './data/glove.6B.50d.json', max_length=max_length, reprocess=False)
test_data_loader = JSONFileDataLoader('./data/test.json', './data/glove.6B.50d.json', max_length=max_length, reprocess=False)

framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)

model = BMAN(train_data_loader.word_vec_mat, max_length, hidden_size=100, args=args)
model_name = args.model_name + str(seed)
framework.train(model, model_name, args.batch, N_for_train=args.N_for_train,  N_for_eval=args.N_for_test,
                K=args.K, Q=args.Q,  learning_rate=args.learning_rate,
                train_iter=50000, val_iter=1000, val_step=2000, test_iter=2000, starttime=0)
