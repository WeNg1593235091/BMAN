import sys

sys.path.append('..')
import torch
from torch import nn
from torch.nn import functional as F
import models.embedding as embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.utils import sort_batch_by_length, init_lstm, init_linear

class BMAN(nn.Module):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=50, pos_embedding_dim=5, args=None,
                 hidden_size=100, drop=True):
        nn.Module.__init__(self)
        self.word_embedding_dim = word_embedding_dim + 2 * pos_embedding_dim
        #隐藏大小
        self.hidden_size = hidden_size
        #最大长度
        self.max_length = max_length
        #嵌入
        self.embedding = embedding.Embedding(word_vec_mat, max_length, word_embedding_dim, pos_embedding_dim)
        self.args = args
        #卷积
        #nn.Conv2d：对由多个输入平面组成的输入信号进行二维卷积。输入通道数、卷积产生的通道数、卷积核尺寸、填充操作
        self.conv = nn.Conv2d(1, self.hidden_size*2, kernel_size=(3, self.word_embedding_dim), padding=(1, 0))
        #nn.Linear（）是用于设置网络中的全连接层的，需要注意在二维图像处理的任务中，全连接层的输入与输出一般都设置为二维张量，形状通常为[batch_size, size]
        '''
        in_features指的是输入的二维张量的大小，即输入的[batch_size, size]中的size。
        out_features指的是输出的二维张量的大小，即输出的二维张量的形状为[batch_size，output_size]，当然，它也代表了该全连接层的神经元个数。
        从输入输出的张量的shape角度来理解，相当于一个输入为[batch_size, in_features]的张量变换成了[batch_size, out_features]的输出张量。
        nn.Linear()：用于设置网络中的全连接层，需要注意的是全连接层的输入与输出都是二维张量。in_features指的是输入的二维张量的大小，即输入的[batch_size, size]中的size。
        out_features指的是输出的二维张量的大小，即输出的二维张量的形状为
        '''
        #全连接层
        self.proj = nn.Linear(self.hidden_size*8, self.hidden_size)
        '''
        nn.LSTM()
        输入的参数列表包括:
        input_size 输入数据的特征维数，通常就是embedding_dim(词向量的维度)
        hidden_size　LSTM中隐层的维度 
        num_layers　循环神经网络的层数
        bias　用不用偏置，default=True
        batch_first 这个要注意，通常我们输入的数据shape=(batch_size,seq_length,embedding_dim),而batch_first默认是False,所以我们的输入数据最好送进LSTM之前将batch_size与seq_length这两个维度调换
        dropout　默认是0，代表不用dropout
        bidirectional默认是false，代表不用双向LSTM
        '''
        self.lstm_enhance = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        #通过Squential将全连接层和激活函数结合起来，输出激活后的网络节点。一个连续的容器。
        self.multilayer = nn.Sequential(nn.Linear(self.hidden_size*8, self.hidden_size),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size, 1))
        self.drop = drop
        #通常意义的dropout解释为：在训练过程的前向传播中，让每个神经元以一定概率p处于不激活的状态。以达到减少过拟合的效果。
        #pytorch中，dropout有另一个用法，这个操作表示使x每个位置的元素都有一定概率归0，以此来模拟现实生活中的某些频道的数据缺失，以达到数据增强的目的。
        ''' 
        nn.dropout()是为了防止或减轻过拟合而使用的函数，它一般用在全连接层
        Dropout就是在不同的训练过程中随机扔掉一部分神经元。也就是让某个神经元的激活值以一定的概率p，让其停止工作，这次训练过程中不更新权值，
        也不参加神经网络的计算。但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了

        '''
        self.dropout = nn.Dropout(0.2)
        #nn.CrossEntropyLoss()为交叉熵损失函数，用于解决多分类问题，也可用于解决二分类问题。在使用nn.CrossEntropyLoss()其内部会自动加上Sofrmax层
        self.cost = nn.CrossEntropyLoss()
        #apply() 函数用于当函数参数已经存在于一个元组或字典中时，间接地调用函数。
        self.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        #Python find() 方法检测字符串中是否包含子字符串 str ，如果指定 beg（开始） 和 end（结束） 范围，则检查是否包含在指定范围内，如果包含子字符串返回开始的索引值，否则返回-1。
        if classname.find('Linear') != -1:
            init_linear(m)
        elif classname.find('LSTM') != -1:
            init_lstm(m)
        # elif classname.find('Conv') != -1:
        #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #     m.weight.data.normal_(0, np.sqrt(2. / n))
        #     if m.bias is not None:
        #         m.bias.data.zero_()

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size.
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        '''
        nn.CrossEntropyLoss()为交叉熵损失函数，用于解决多分类问题，也可用于解决二分类问题。在使用nn.CrossEntropyLoss()其内部会自动加上Sofrmax层
        第一个参数：x为输入也是网络的最后一层的输出，其shape为[batchsize,class]（函数要求第一个参数，也就是最后一层的输出为二维数据，每个向量中的值为不同种类的概率值），如果batchsize大于1，那么求对应的均值。
        第二个参数：是传入的标签，也就是某个类别的索引值
        https://blog.csdn.net/weixin_45414792/article/details/120778065
        '''
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        #view(-1)将张量设置为一维
        return torch.mean((pred.view(-1) == label.view(-1)).float())
    #上下文编码器
    def context_encoder(self, input):
        input_mask = (input['mask'] != 0).float()  # float() 函数用于将整数和字符串转换成浮点数。
        '''
        sum(1)求数组中每一行的和
        max() 方法返回给定参数的最大值，参数可以为序列。
        item（）取出单元素张量的元素值并返回该值，保持原元素类型不变
        调用contiguous()之后，PyTorch会开辟一块新的内存空间存放变换之后的数据
        '''
        max_length = input_mask.long().sum(1).max().item()
        input_mask = input_mask[:, :max_length].contiguous()
        embedding = self.embedding(input)
        embedding_ = embedding[:, :max_length].contiguous()

        if self.drop:
            embedding_ = self.dropout(embedding_)
        '''
        unsqueeze(i)表示用一个箱子把第i层的箱子都包起来；
        而squeeze(i)表示把第i层的箱子去掉（第i层只有一个箱子时才能用这个函数）

        '''
        conv_out = self.conv(embedding_.unsqueeze(1)).squeeze(3)
        conv_out = conv_out * input_mask.unsqueeze(1)  # 运算符*在矩阵运算中的功能是逐元素的乘法（称为Hadamard积（Hadamard product，数学符号⊙））。

        return conv_out.transpose(1, 2).contiguous(), input_mask, max_length

    def lstm_encoder(self, input, mask, lstm):#enhance_support, support_mask, self.lstm_enhance
        if self.drop:
            input = self.dropout(input)
        mask = mask.squeeze(2)
        sequence_lengths = mask.long().sum(1)
        sorted_inputs, sorted_sequence_lengths, restoration_indices, _ = sort_batch_by_length(input, sequence_lengths)
        ''' 
        pack之后，原来填充的 PAD（一般初始化为0）占位符被删掉了。函数的功能是将一个填充后的变长序列压紧。

        输入的形状可以是(T×B×* )。T是最长序列长度，B是batch size，*代表任意维度(可以是0)。如果batch_first=True的话，那么相应的 input size 就是 (B×T×*)。

        Variable中保存的序列，应该按序列长度的长短排序，长的在前，短的在后。即input[:,0]代表的是最长的序列，input[:, B-1]保存的是最短的序列。

        NOTE： 只要是维度大于等于2的input都可以作为这个函数的参数。你可以用它来打包labels，然后用RNN的输出和打包后的labels来计算loss。通过PackedSequence对象的.data属性可以获取 Variable。

        参数说明:

        input (Variable) – 变长序列 被填充后的 batch
        lengths (list[int]) – Variable 中 每个序列的长度。
        batch_first (bool, optional) – 如果是True，input的形状应该是B*T*size。
        返回值:

        一个PackedSequence 对象。
        '''
        packed_sequence_input = pack_padded_sequence(sorted_inputs,
                                                     sorted_sequence_lengths.to("cpu"),
                                                     batch_first=True)
        lstmout, _ = lstm(packed_sequence_input)
        ''' 
        pad_packed_sequence()
        这个操作和pack_padded_sequence()是相反的。把压紧的序列再填充回来。填充时会初始化为0。

        返回的Varaible的值的size是 T×B×*, T 是最长序列的长度，B 是 batch_size,如果 batch_first=True,那么返回值是B×T×*。

        Batch中的元素将会以它们长度的逆序排列。

        参数说明:

        sequence (PackedSequence) – 将要被填充的 batch
        batch_first (bool, optional) – 如果为True，返回的数据的格式为 B×T×*。
        返回值: 一个tuple，包含被填充后的序列，和batch中序列的长度列表
        '''
        unpacked_sequence_tensor, _ = pad_packed_sequence(lstmout, batch_first=True)
        unpacked_sequence_tensor = unpacked_sequence_tensor.index_select(0, restoration_indices)

        return unpacked_sequence_tensor


    def CoAttention(self, support, query, support_mask, query_mask):
        '''
        @:矩阵相乘
        transpose：矩阵转置
        运算符*在矩阵运算中的功能是逐元素的乘法（称为Hadamard积（Hadamard product，数学符号⊙））。
        矩阵*数字矩阵中的数字都翻倍
        F.softmax(score, dim=1)
        dim=1就是对score矩阵中 所有第1维下标不同，其他维下标均相同的元素进行操作（softmax）操作后维度上的元素相加等于1
        '''
        att = support @ query.transpose(1, 2)
        att = att + support_mask * query_mask.transpose(1, 2) * 100
        support_ = F.softmax(att, 2) @ query * support_mask
        query_ = F.softmax(att.transpose(1,2), 2) @ support * query_mask
        #匹配表示
        return support_, query_

    def local_matching(self, support, query, support_mask, query_mask):

        support_, query_ = self.CoAttention(support, query, support_mask, query_mask)
        enhance_query = self.fuse(query, query_, 2)
        enhance_support = self.fuse(support, support_, 2)

        return enhance_support, enhance_query

    def fuse(self, m1, m2, dim):
        '''
        torch.abs绝对值
        C=torch.cat((A,B),1)就表示按维数1（列）拼接A和B，也就是横着拼接，A左B右。
        '''
        return torch.cat([m1, m2, torch.abs(m1 - m2), m1 * m2], dim)

    def local_aggregation(self, enhance_support, enhance_query, support_mask, query_mask, K):

        max_enhance_query, _ = torch.max(enhance_query, 1)
        mean_enhance_query = torch.sum(enhance_query, 1) / torch.sum(query_mask, 1)
        enhance_query = torch.cat([max_enhance_query, mean_enhance_query], 1)

        enhance_support = enhance_support.view(enhance_support.size(0) // K, K, -1, self.hidden_size * 2)
        support_mask = support_mask.view(enhance_support.size(0), K, -1, 1)

        max_enhance_support, _ = torch.max(enhance_support, 2)
        mean_enhance_support = torch.sum(enhance_support, 2) / torch.sum(support_mask, 2)
        enhance_support = torch.cat([max_enhance_support, mean_enhance_support], 2)

        return enhance_support, enhance_query

    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''
        #print(111111110)
        support, support_mask, support_len = self.context_encoder(support)
        query, query_mask, query_len = self.context_encoder(query)

        batch = support.size(0)//(N*K) #取整 N*K：支持集中实例的个数
        # concate S_k operation
        '''
        expand()其将单个维度扩大成更大维度，返回一个新的tensor，通过expand（）函数扩展某一维度后自身不会发生变化
        view()该函数返回一个有__相同数据__但不同大小的 Tensor。通俗一点，就是__改变矩阵维度_
        '''
        support = support.view(batch, 1, N, K, support_len, self.hidden_size*2).expand(batch, N*Q, N, K, support_len, self.hidden_size*2).contiguous().view(batch*N*Q*N, K*support_len, self.hidden_size*2)
        support_mask = support_mask.view(batch, 1, N, K, support_len).expand(batch, N*Q, N, K, support_len).contiguous().view(-1, K*support_len, 1)
        query = query.view(batch, N*Q, 1, query_len, self.hidden_size*2).expand(batch, N*Q, N, query_len, self.hidden_size*2).contiguous().view(batch*N*Q*N, query_len, self.hidden_size*2)
        query_mask = query_mask.view(batch, N*Q, 1, query_len).expand(batch, N*Q, N, query_len).contiguous().view(-1, query_len, 1)

        enhance_support, enhance_query = self.local_matching(support, query, support_mask, query_mask)

        # reduce dimensionality 降维
        enhance_support = self.proj(enhance_support)  #*, self.hidden_size
        enhance_query = self.proj(enhance_query)
        #激活函数
        enhance_support = torch.relu(enhance_support) #nn.ReLU(inplace=True) inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出
        enhance_query = torch.relu(enhance_query)

        # split operation 拆分操作
        ''' 
        view()相当于reshape、resize，重新调整Tensor的形状。
        '''
        enhance_support = enhance_support.view(batch*N*Q*N*K, support_len, self.hidden_size)
        support_mask = support_mask.view(batch*N*Q*N*K, support_len, 1)

        # LSTM
        enhance_support = self.lstm_encoder(enhance_support, support_mask, self.lstm_enhance)
        enhance_query = self.lstm_encoder(enhance_query, query_mask, self.lstm_enhance)

        # Local aggregation 局部聚集

        enhance_support, enhance_query = self.local_aggregation(enhance_support, enhance_query, support_mask, query_mask, K)
        #即repeat的参数是对应维度的复制个数，上段代码为0维复制两次，1维复制两次，则得到以上运行结果
        tmp_query = enhance_query.unsqueeze(1).repeat(1, K, 1)
        #torch.cat(inputs, dimension=0) 在给定维度上对输入的张量序列seq 进行连接操作。
        #参数:
        #inputs (sequence of Tensors) – 可以是任意相同Tensor 类型的python 序列
        #dimension (int, optional) – 沿着此维连接张量序列。
        cat_seq = torch.cat([tmp_query, enhance_support], 2)
        beta = self.multilayer(cat_seq) #查询集与支持实例的实例级匹配程度
        one_enhance_support = (enhance_support.transpose(1, 2) @ F.softmax(beta, 1)).squeeze(2)# 关系原型
        ''' 
        １．torch.sum(input, dtype=None)
        ２．torch.sum(input, list: dim, bool: keepdim=False, dtype=None) → Tensor
　
         input:输入一个tensor
         dim:要求和的维度，可以是一个列表
         keepdim:求和之后这个dim的元素个数为１，所以要被去掉，如果要保留这个维度，则应当keepdim=True

        '''
        J_incon = torch.sum((one_enhance_support.unsqueeze(1) - enhance_support) ** 2, 2).mean()#用支持实例与类原型之间的平均欧式距离来计算不一致性度量

        cat_seq = torch.cat([enhance_query, one_enhance_support], 1)
        logits = self.multilayer(cat_seq) #类级匹配

        logits = logits.view(batch*N*Q, N)
        ''' 
        input是一个softmax函数输出的tensor
       （当然，如果用CrossEntropyLoss()，最后一层是一个全连接层）
        dim是max函数索引的维度 0或1 0是每列的最大值，1是每行的最大值
        此函数输出两个tensor
        1.第一个tensor是每行的最大概率
        2.第二个tensor是每行最大概率的索引
        由于我们不需要获取最大概率的值，只要知道最大概率的是哪个类别即可
        因此，我们只需要获取第二个tensor
        '''
        #_, pred = torch.max(logits, 1)

        #return logits, pred, J_incon
        return logits, J_incon