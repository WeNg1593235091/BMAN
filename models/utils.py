import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn

def sort_batch_by_length(tensor: torch.Tensor, sequence_lengths: torch.Tensor):
    """
    Sort a batch first tensor by some specified lengths.按指定的长度对批处理第一张量排序
    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A batch first Pytorch tensor.
    sequence_lengths : torch.LongTensor, required.
        A tensor representing the lengths of some dimension of the tensor which表示我们想要排序的张量的某个维度的长度。
        we want to sort by.
    Returns
    -------
    LongTensor,FloatTensor等，都是创建相对应的数据类型；
    sorted_tensor : torch.FloatTensor
        The original tensor sorted along the batch dimension with respect to sequence_lengths.原始张量沿着批次维度按照序列长度排序。
    sorted_sequence_lengths : torch.LongTensor
        The original sequence_lengths sorted by decreasing size.原始序列长度按大小递减排序。
    restoration_indices : torch.LongTensor
        Indices into the sorted_tensor such that
        ``sorted_tensor.index_select(0, restoration_indices) == original_tensor``
        注意：sorted_tensor.index_select(0, restoration_indices) == original_tensor 说明
        这种方式可以将排序的数据恢复到原数据
    permuation_index : torch.LongTensor
        The indices used to sort the tensor. This is useful if you want to sort many
        tensors using the same ordering.
    """
    ''' 
    torch.sort：对输入数据排序，返回两个值，即排序后的数据values和其在原矩阵中的坐标indices
    input：输入矩阵
    dim：排序维度，默认为dim=1,即对行排序
    descending：排序方式（从小到大和从大到小），默认为从小到大排序（即descending=False)
    def index_select(self, dim: _int, index: Tensor)
    第二个参数dim表示维度，具体取哪一个维度就要看你对象原本的维度数了，比如一个torch.Size([1, 3, 3])的张量，
    你想取某一列，那dim就该取2或-1，一个torch.Size([ 3,3])的张量，想取某一行的话，那dim就该取1或-1。

     第三个参数index表示你想取的索引的序号，比如torch.tensor([0, 1])就是取第一第二列。

    '''
    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)
    ''' 
    torch.autograd.Variable是Autograd的核心类，它封装了Tensor，并整合了反向传播的相关实现
    Varibale包含三个属性：

    data：存储了Tensor，是本体的数据
    grad：保存了data的梯度，本事是个Variable而非Tensor，与data形状一致
    grad_fn：指向Function对象，用于反向传播的梯度计算之用
    '''
    index_range = Variable(torch.arange(0, len(sequence_lengths)).long()).cuda()

    # This is the equivalent of zipping with index, sorting by the original
    # sequence lengths and returning the now sorted indices.
    #这相当于使用索引压缩，按原始序列长度排序，然后返回现在排序的索引。
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index

def init_linear(input_linear):
    """
    初始化线性变换
    Initialize linear transformation
    """
    '''
    size()函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数。
    np.sqrt(B):求B的开方（算数平方根）
    '''
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    '''
    torch.nn.init.uniform(tensor, a=0, b=1)
    从均匀分布U(a, b)中生成值，填充输入的张量或变量

    参数：
    tensor - n维的torch.Tensor
    a - 均匀分布的下界
    b - 均匀分布的上界
    '''
    nn.init.uniform_(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        #清零
        input_linear.bias.data.zero_()

def init_lstm(input_lstm):
    """
    Initialize lstm
    """
    '''
    python2.x range() 函数可创建一个整数列表，一般用在 for 循环中。
    Python3 range() 返回的是一个可迭代对象（类型是对象），而不是列表类型， 所以打印的时候不会打印列表，
    range(start, stop[, step])
    start: 计数从 start 开始。默认是从 0 开始。例如range（5）等价于range（0， 5）;
    stop: 计数到 stop 结束，但不包括 stop。例如：range（0， 5） 是[0, 1, 2, 3, 4]没有5
    step：步长，默认为1。例如：range（0， 5） 等价于 range(0, 5, 1)
    '''
    for ind in range(0, input_lstm.num_layers):
        #eval() 函数用来执行一个字符串表达式，并返回表达式的值。str() 函数将对象转化为适于人阅读的形式。
        #~LSTM.weight_ih_l[k] – 学习得到的第k层的 input-hidden 权重
        weight = eval('input_lstm.weight_ih_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform_(weight, -bias, bias)
        #~LSTM.weight_hh_l[k] –学习得到的第k层的 hidden -hidden 权重  y
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform_(weight, -bias, bias)
    # if input_lstm.bidirectional:
    #     for ind in range(0, input_lstm.num_layers):
    #         weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
    #         bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
    #         nn.init.uniform(weight, -bias, bias)
    #         weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
    #         bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
    #         nn.init.uniform(weight, -bias, bias)

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            #~LSTM.bias_ih_l[k] – 学习得到的第k层的input-hidden 的偏置
            weight = eval('input_lstm.bias_ih_l' + str(ind))
            # 清零
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            #~LSTM.bias_hh_l[k] – 学习得到的第k层的hidden -hidden 的偏置
            weight = eval('input_lstm.bias_hh_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
        # if input_lstm.bidirectional:
        #     for ind in range(0, input_lstm.num_layers):
        #         weight = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
        #         weight.data.zero_()
        #         weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
        #         weight = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
        #         weight.data.zero_()
        #         weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

def init_cnn(input_cnn):
    n = input_cnn.in_channels
    for k in input_cnn.kernel_size:
        n *= k
    stdv = np.sqrt(6./n)
    input_cnn.weight.data.uniform_(-stdv, stdv)
    if input_cnn.bias is not None:
        input_cnn.bias.data.uniform_(-stdv, stdv)
