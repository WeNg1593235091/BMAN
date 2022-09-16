import json
import os
import numpy as np
import random
import torch

class JSONFileDataLoader:
    def _load_preprocessed_file(self):
        #print(self)
        #print(self.file_name)
        #print(self.word_vec_file_name)
        name_prefix = '.'.join(self.file_name.split('/')[-1].split('.')[:-1])        #字符串拆分函数.split(),字符串拼接函数.join(),[-1]去最后一个元素，[:-1]除了最后一个取全部,[::-1]取从后向前（相反）的元素
        word_vec_name_prefix = '.'.join(self.word_vec_file_name.split('/')[-1].split('.')[:-1])
        processed_data_dir = '_processed_data'
        if not os.path.isdir(processed_data_dir):                      #判断对象是否为一个目录
            return False                                              #不存在就返回False
        word_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_word.npy')  #链接两个或多个路径名组件
        print(word_npy_file_name)
        pos1_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_pos1.npy')
        pos2_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_pos2.npy')
        mask_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_mask.npy')
        length_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_length.npy')
        rel2scope_file_name = os.path.join(processed_data_dir, name_prefix + '_rel2scope.json')
        word_vec_mat_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy')
        word2id_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json')
        if not os.path.exists(word_npy_file_name) or \
           not os.path.exists(pos1_npy_file_name) or \
           not os.path.exists(pos2_npy_file_name) or \
           not os.path.exists(mask_npy_file_name) or \
           not os.path.exists(length_npy_file_name) or \
           not os.path.exists(rel2scope_file_name) or \
           not os.path.exists(word_vec_mat_file_name) or \
           not os.path.exists(word2id_file_name):
            return False
        print("Pre-processed files exist. Loading them...")
        self.data_word = np.load(word_npy_file_name) #读取npy文件
        self.data_pos1 = np.load(pos1_npy_file_name)
        self.data_pos2 = np.load(pos2_npy_file_name)
        self.data_mask = np.load(mask_npy_file_name)
        self.data_length = np.load(length_npy_file_name)
        self.rel2scope = json.load(open(rel2scope_file_name))#读取json文件
        self.word_vec_mat = np.load(word_vec_mat_file_name)
        self.word2id = json.load(open(word2id_file_name))
        if self.data_word.shape[1] != self.max_length:   #矩阵的列不等于最大长度
            print("Pre-processed files don't match current settings. Reprocessing...")
            return False
        print("Finish loading")
        return True

    def __init__(self, file_name, word_vec_file_name, max_length=40, case_sensitive=False, reprocess=False):
        '''
        file_name: Json file storing the data in the following format
            {
                "P155": # relation id
                    [
                        {
                            "h": ["song for a future generation", "Q7561099", [[16, 17, ...]]], # head entity [word, id, location]
                            "t": ["whammy kiss", "Q7990594", [[11, 12]]], # tail entity [word, id, location]
vugu                        },
                        ...
                    ],
                "P177": 
                    [
                        ...
                    ]
                ...
            }
        word_vec_file_name: Json file storing word vectors in the following format
            [
                {'word': 'the', 'vec': [0.418, 0.24968, ...]},
                {'word': ',', 'vec': [0.013441, 0.23682, ...]},
                ...
            ]
        max_length: The length that all the sentences need to be extend to.
        case_sensitive: Whether the data processing is case-sensitive, default as False.
        reprocess: Do the pre-processing whether there exist pre-processed files, default as False.
        '''
        #print(file_name)
        self.file_name = file_name
        #print(self.file_name)
        #print(word_vec_file_name)
        self.word_vec_file_name = word_vec_file_name
        self.case_sensitive = case_sensitive
        self.max_length = max_length

        if reprocess or not self._load_preprocessed_file(): # Try to load pre-processed files:
            # Check files
            if file_name is None or not os.path.isfile(file_name):
                raise Exception("[ERROR] Data file doesn't exist")
            if word_vec_file_name is None or not os.path.isfile(word_vec_file_name):
                raise Exception("[ERROR] Word vector file doesn't exist")

            # Load files
            print("Loading data file...")
            #json.load(f)之后，返回的对象是python的字典对象
            self.ori_data = json.load(open(self.file_name, "r"))
            print("Finish loading")
            print("Loading word vector file...")
            self.ori_word_vec = json.load(open(self.word_vec_file_name, "r"))
            print("Finish loading")
            
            # Eliminate case sensitive
            if not case_sensitive:
                print("Elimiating case sensitive problem...")
                for relation in self.ori_data:
                    for ins in self.ori_data[relation]:
                        #Python len() 方法返回对象（字符、列表、元组等）长度或项目个数。
                        for i in range(len(ins['tokens'])):
                            #Python lower() 方法转换字符串中所有大写字符为小写。
                            ins['tokens'][i] = ins['tokens'][i].lower()
                print("Finish eliminating")

            # Pre-process word vec
            self.word2id = {}
            self.word_vec_tot = len(self.ori_word_vec)  #len()获取词向量文件序列长度
            print(self.word_vec_tot)
            UNK = self.word_vec_tot
            BLANK = self.word_vec_tot + 1
            self.word_vec_dim = len(self.ori_word_vec[0]['vec'])
            print(self.word_vec_dim)
            print("Got {} words of {} dims".format(self.word_vec_tot, self.word_vec_dim))
            print("Building word vector matrix and mapping...")
            self.word_vec_mat = np.zeros((self.word_vec_tot, self.word_vec_dim), dtype=np.float32) #返回一个元素全为0且给定形状和类型的数组
            for cur_id, word in enumerate(self.ori_word_vec): #enumerate()枚举 glove.6B.50d.json
                w = word['word']
                if not case_sensitive:
                    w = w.lower()
                self.word2id[w] = cur_id  #glove.6B.50d_word2id.json
                #word_vec_mat是一个矩阵，cur_id是矩阵中的行，将每个词的向量赋值到词矩阵对应的行
                self.word_vec_mat[cur_id, :] = word['vec']
                #np.sqrt(B):求B的开方（算数平方根）
                self.word_vec_mat[cur_id] = self.word_vec_mat[cur_id] / np.sqrt(np.sum(self.word_vec_mat[cur_id] ** 2))


            self.word2id['UNK'] = UNK
            self.word2id['BLANK'] = BLANK
            print("Finish building")

            # Pre-process data
            print("Pre-processing data...")
            self.instance_tot = 0
            for relation in self.ori_data:
                self.instance_tot += len(self.ori_data[relation])#每一个类中的实例数
            self.data_word = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)  #zeros返回一个给定形状和类型的用0填充的数组
            self.data_pos1 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_pos2 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_mask = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_length = np.zeros((self.instance_tot), dtype=np.int32)
            self.rel2scope = {} # left closed and right open
            i = 0
            for relation in self.ori_data:
                self.rel2scope[relation] = [i, i]
                for ins in self.ori_data[relation]:
                    pos1 = ins['h'][2][0][0]
                    pos2 = ins['t'][2][0][0]
                    words = ins['tokens']
                    cur_ref_data_word = self.data_word[i]         
                    for j, word in enumerate(words):
                        if j < max_length:
                            if word in self.word2id:
                                cur_ref_data_word[j] = self.word2id[word]#为每一个实例中的词编码
                            else:
                                cur_ref_data_word[j] = UNK
                        else:
                            break
                    for j in range(j + 1, max_length):
                        cur_ref_data_word[j] = BLANK
                    self.data_length[i] = len(words) #实例中词的个数
                    if len(words) > max_length:
                        self.data_length[i] = max_length
                    if pos1 >= max_length:
                        pos1 = max_length - 1
                    if pos2 >= max_length:
                        pos2 = max_length - 1
                    pos_min = min(pos1, pos2)
                    pos_max = max(pos1, pos2)
                    for j in range(max_length):
                        self.data_pos1[i][j] = j - pos1 + max_length
                        self.data_pos2[i][j] = j - pos2 + max_length
                        if j >= self.data_length[i]:
                            self.data_mask[i][j] = 0
                            self.data_pos1[i][j] = 0
                            self.data_pos2[i][j] = 0
                        elif j <= pos_min:
                            self.data_mask[i][j] = 1
                        elif j <= pos_max:
                            self.data_mask[i][j] = 2
                        else:
                            self.data_mask[i][j] = 3
                    i += 1
                self.rel2scope[relation][1] = i #train_rel2scope.json

            print("Finish pre-processing")
            print("Storing processed files...")
            name_prefix = '.'.join(file_name.split('/')[-1].split('.')[:-1])
            word_vec_name_prefix = '.'.join(word_vec_file_name.split('/')[-1].split('.')[:-1])
            processed_data_dir = '_processed_data'
            if not os.path.isdir(processed_data_dir): #判断processed_data_dir是否为路径
                os.mkdir(processed_data_dir)  #创建目录
            '''
            np.save(file, arr, allow_pickle=True, fix_imports=True)
            解释：Save an array to a binary file in NumPy .npy format。以“.npy”格式将数组保存到二进制文件中。
            参数：
            file 要保存的文件名称，需指定文件保存路径，如果未设置，保存到默认路径。其文件拓展名为.npy
            arr 为需要保存的数组，也即把数组arr保存至名称为file的文件中。

            '''
            np.save(os.path.join(processed_data_dir, name_prefix + '_word.npy'), self.data_word)#np.save存储数组数据
            np.save(os.path.join(processed_data_dir, name_prefix + '_pos1.npy'), self.data_pos1)
            np.save(os.path.join(processed_data_dir, name_prefix + '_pos2.npy'), self.data_pos2)
            np.save(os.path.join(processed_data_dir, name_prefix + '_mask.npy'), self.data_mask)
            np.save(os.path.join(processed_data_dir, name_prefix + '_length.npy'), self.data_length)
            json.dump(self.rel2scope, open(os.path.join(processed_data_dir, name_prefix + '_rel2scope.json'), 'w')) #编写json文件
            np.save(os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy'), self.word_vec_mat)
            json.dump(self.word2id, open(os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json'), 'w'))
            print("Finish storing")

    def next_one(self, N=5, K=5, Q=100):
        '''
        Python 字典(Dictionary) keys() 函数以列表返回一个字典所有的键。
        sample(序列a，n)
        功能：从序列a中随机抽取n个元素，并将n个元素生以list形式返回。
        N: Num of classes for each batch 20
        '''
        target_classes = random.sample(self.rel2scope.keys(), N)  #截取指定长度的随机数,种类的个数。选出的个数为N,但是内容是随机的
        #print(target_classes)
        #'''
        #正向
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_label = []
        #'''
        #逆向
        support_set_n = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_set_n = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_label_n = []



        for i, class_name in enumerate(target_classes): #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            #范围
            scope = self.rel2scope[class_name]
            '''
              list() 方法用于将元组转换为列表。
            #numpy.random.choice(a, size=None, replace=True, p=None)
            #从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
            #replace:True表示可以取相同数字，False表示不可以取相同数字
            #数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
            K: Num of instances for each class in the support set
            Q: Num of instances for each class in the query set         
            '''
            indices = np.random.choice(list(range(scope[0], scope[1])), K + Q, False) #从一维的数据中随机抽取数字并组成指定大小的数组
            word = self.data_word[indices]
            pos1 = self.data_pos1[indices]
            pos2 = self.data_pos2[indices]
            mask = self.data_mask[indices]
            #正向
           # '''
            support_word, query_word = np.split(word, [K]) #把一个数组从左到右按顺序切分
            support_pos1, query_pos1 = np.split(pos1, [K])
            support_pos2, query_pos2 = np.split(pos2, [K])
            support_mask, query_mask = np.split(mask, [K])
           # '''
            #逆向
            query_word_n, support_word_n = np.split(word, [Q])
            query_pos1_n, support_pos1_n = np.split(pos1, [Q])
            query_pos2_n, support_pos2_n = np.split(pos2, [Q])
            query_mask_n, support_mask_n = np.split(mask, [Q])
           # '''
            #正向
            support_set['word'].append(support_word) #在列表后添加对象
            support_set['pos1'].append(support_pos1)
            support_set['pos2'].append(support_pos2)
            support_set['mask'].append(support_mask)
            query_set['word'].append(query_word)
            query_set['pos1'].append(query_pos1)
            query_set['pos2'].append(query_pos2)
            query_set['mask'].append(query_mask)
            query_label += [i] * Q #果用一个列表list1乘一个数字n 会得到一个新的列表list2, 这个列表的元素是list1的元素重复n次；列表相加就是列表按顺序合并
           # '''
            #逆向
            support_set_n['word'].append(support_word_n)  # 在列表后添加对象
            support_set_n['pos1'].append(support_pos1_n)
            support_set_n['pos2'].append(support_pos2_n)
            support_set_n['mask'].append(support_mask_n)
            query_set_n['word'].append(query_word_n)
            query_set_n['pos1'].append(query_pos1_n)
            query_set_n['pos2'].append(query_pos2_n)
            query_set_n['mask'].append(query_mask_n)
            query_label_n += [i] * Q
       # '''
        #正向
        support_set['word'] = np.stack(support_set['word'], 0)#从0维堆叠数组
        support_set['pos1'] = np.stack(support_set['pos1'], 0)
        support_set['pos2'] = np.stack(support_set['pos2'], 0)
        support_set['mask'] = np.stack(support_set['mask'], 0)
        query_set['word'] = np.concatenate(query_set['word'], 0)#按照列不变拼接
        query_set['pos1'] = np.concatenate(query_set['pos1'], 0)
        query_set['pos2'] = np.concatenate(query_set['pos2'], 0)
        query_set['mask'] = np.concatenate(query_set['mask'], 0)
        query_label = np.array(query_label) #将列表转化为数组
       # '''
        #逆向
        support_set_n['word'] = np.stack(support_set_n['word'], 0)  # 从0维堆叠数组
        support_set_n['pos1'] = np.stack(support_set_n['pos1'], 0)
        support_set_n['pos2'] = np.stack(support_set_n['pos2'], 0)
        support_set_n['mask'] = np.stack(support_set_n['mask'], 0)
        query_set_n['word'] = np.concatenate(query_set_n['word'], 0)  # 按照列不变拼接
        query_set_n['pos1'] = np.concatenate(query_set_n['pos1'], 0)
        query_set_n['pos2'] = np.concatenate(query_set_n['pos2'], 0)
        query_set_n['mask'] = np.concatenate(query_set_n['mask'], 0)
        query_label_n = np.array(query_label_n)

        # return support_set, query_set, query_label
        #return support_set_n, query_set_n, query_label_n
        return support_set, query_set, query_label, support_set_n, query_set_n, query_label_n


    # B: Batch size
    def next_batch(self, B=4, N=20, K=5, Q=100):
        #'''
        #正向
        support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        label = []
        #'''
        #逆向
        support_n = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_n = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        label_n = []
        for one_sample in range(B):
            current_support, current_query, current_label, current_support_n, current_query_n, current_label_n = self.next_one(N, K, Q)
            #'''
            #正向
            #current_support, current_query, current_label = self.next_one(N, K, Q)
            support['word'].append(current_support['word'])#用于在列表末尾添加新的对象
            support['pos1'].append(current_support['pos1'])
            support['pos2'].append(current_support['pos2'])
            support['mask'].append(current_support['mask'])
            query['word'].append(current_query['word'])
            query['pos1'].append(current_query['pos1'])
            query['pos2'].append(current_query['pos2'])
            query['mask'].append(current_query['mask'])
            #在列表末尾添加新的对象
            label.append(current_label)
            #'''
            #逆向
            #current_support_n, current_query_n, current_label_n = self.next_one(N, K, Q)
            support_n['word'].append(current_support_n['word'])  # 用于在列表末尾添加新的对象
            support_n['pos1'].append(current_support_n['pos1'])
            support_n['pos2'].append(current_support_n['pos2'])
            support_n['mask'].append(current_support_n['mask'])
            query_n['word'].append(current_query_n['word'])
            query_n['pos1'].append(current_query_n['pos1'])
            query_n['pos2'].append(current_query_n['pos2'])
            query_n['mask'].append(current_query_n['mask'])
            label_n.append(current_label_n)
        '''
        stack(w,w)从维堆叠数组
        torch.from_numpy将数组转化为张量 
        long()将数字或字符串转换为一个长整型。     
        view(-1，self.max_length）固定张量的形状
        '''
        #'''
        #正向
        support['word'] = torch.from_numpy(np.stack(support['word'], 0)).long().view(-1, self.max_length)
        support['pos1'] = torch.from_numpy(np.stack(support['pos1'], 0)).long().view(-1, self.max_length)
        support['pos2'] = torch.from_numpy(np.stack(support['pos2'], 0)).long().view(-1, self.max_length)
        support['mask'] = torch.from_numpy(np.stack(support['mask'], 0)).long().view(-1, self.max_length)
        query['word'] = torch.from_numpy(np.stack(query['word'], 0)).long().view(-1, self.max_length)
        query['pos1'] = torch.from_numpy(np.stack(query['pos1'], 0)).long().view(-1, self.max_length)
        query['pos2'] = torch.from_numpy(np.stack(query['pos2'], 0)).long().view(-1, self.max_length)
        query['mask'] = torch.from_numpy(np.stack(query['mask'], 0)).long().view(-1, self.max_length)
        #torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变
        label = torch.from_numpy(np.stack(label, 0).astype(np.int64)).long()
        #'''
        #逆向
        support_n['word'] = torch.from_numpy(np.stack(support_n['word'], 0)).long().view(-1, self.max_length)
        support_n['pos1'] = torch.from_numpy(np.stack(support_n['pos1'], 0)).long().view(-1, self.max_length)
        support_n['pos2'] = torch.from_numpy(np.stack(support_n['pos2'], 0)).long().view(-1, self.max_length)
        support_n['mask'] = torch.from_numpy(np.stack(support_n['mask'], 0)).long().view(-1, self.max_length)
        query_n['word'] = torch.from_numpy(np.stack(query_n['word'], 0)).long().view(-1, self.max_length)
        query_n['pos1'] = torch.from_numpy(np.stack(query_n['pos1'], 0)).long().view(-1, self.max_length)
        query_n['pos2'] = torch.from_numpy(np.stack(query_n['pos2'], 0)).long().view(-1, self.max_length)
        query_n['mask'] = torch.from_numpy(np.stack(query_n['mask'], 0)).long().view(-1, self.max_length)
        label_n = torch.from_numpy(np.stack(label_n, 0).astype(np.int64)).long()

        #'''
        for key in support:
            support[key] = support[key].cuda()
        for key in query:
            query[key] = query[key].cuda()
        label = label.cuda()
        #'''
        for key in support_n:
            support_n[key] = support_n[key].cuda()
        for key in query_n:
            query_n[key] = query_n[key].cuda()
        label_n = label_n.cuda()

        return support, query, label, support_n, query_n, label_n
