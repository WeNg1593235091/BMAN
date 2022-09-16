import datetime
import os
import torch
from torch import optim


class FewShotREFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader):

        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader

    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict     #返回点指令
        '''
        if os.path.isfile(ckpt):  # 判断提供的绝对路径是否为文件
            checkpoint = torch.load(ckpt)  # 从本地模型中读取数据
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def train(self, model, model_name, B=4, N_for_train=20, N_for_eval=5, K=5, Q=100,
              ckpt_dir='./checkpoint', learning_rate=1e-1, lr_step_size=20000,
              weight_decay=1e-5, train_iter=30000, val_iter=1000, val_step=2000,
              test_iter=3000, pretrain_model=None, optimizer=optim.SGD, starttime=0):
        '''
        model: model
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints    #检查点目录
        test_result_dir: Directory of test results   #测试结果目录
        learning_rate: Initial learning rate    #初始学习率
        lr_step_size: Decay learning rate every lr_step_size steps  #每步衰减学习率
        weight_decay: Rate of decaying weight   #权重衰减率
        train_iter: Num of iterations of training #训练的迭代次数
        val_iter: Num of iterations of validating #验证的迭代次数
        val_step: Validate every val_step steps #验证每一val_step的步数
        test_iter: Num of iterations of testing #测试的迭代次数
        cuda: Use CUDA or not
        pretrain_model: Pre-trained checkpoint path #预先训练的检查点路径
        '''
        # Init
        '''
                filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回一个迭代器对象，如果要转换为列表，可以使用 list() 来转换。
                         该接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，然后返回 True 或
                         False，最后将返回 True 的元素放到新列表中。
                所有的tensor都有.requires_grad属性,可以设置这个属性
                parameters里存的就是weight，parameters()会返回一个生成器（迭代器）
                requires_grad是Pytorch中通用数据结构Tensor的一个属性，用于说明当前量是否需要在计算中保留对应的梯度信息,https://blog.csdn.net/weixin_44696221/article/details/104269981
                lambda本质上是个函数功能，是个匿名的函数，表达形式和用法均与一般函数有所不同。普通的函数可以写简单的也可以写复杂的，但lambda函数一般在一行内实现，是个非常简单的函数功能体。
                '''
        # 优化参数
        parameters_to_optimize = filter(lambda x: x.requires_grad, model.parameters())  # 用于过滤序列，过滤掉不符合条件的元素。
        # 优化
        optimizer = optimizer(parameters_to_optimize, learning_rate, weight_decay=weight_decay)
        # 调度程序
        '''
                每过step_size个epoch，做一次更新
                optimizer （Optimizer）：要更改学习率的优化器；
                step_size（int）：每训练step_size个epoch，更新一次参数；
         '''
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)  # 每过step_size个epoch对optimizer的学习率更新一次
        if pretrain_model:
            checkpoint = self.__load_model__(pretrain_model)
            '''
                        load_state_dict(state_dict, strict=True)

                        从 state_dict 中复制参数和缓冲区到 Module 及其子类中 

                        state_dict：包含参数和缓冲区的 Module 状态字典

                        strict：默认 True，是否严格匹配 state_dict 的键值和 Module.state_dict()的键值

            '''
            model.load_state_dict(checkpoint['state_dict'])  # 将预训练的参数权重加载到新的模型之中
            start_iter = checkpoint['iter'] + 1
        else:
            start_iter = 0

        model = model.cuda()  # 将模型加载到GPU上
        model.train()

        # Training
        best_acc = 0
        for it in range(start_iter, start_iter + train_iter):  # 创建一个整数列表［start_iter,s+t）
            '''
                       在scheduler的step_size表示scheduler.step()每调用step_size次，对应的学习率就会按照策略调整一次。
                       所以如果scheduler.step()是放在mini-batch里面，那么step_size指的是经过这么多次迭代，学习率改变一次。
                       B=4, N_for_train=20, N_for_eval=5, K=5, Q=100
                       B: Batch size
                       N: Num of classes for each batch
                       K: Num of instances for each class in the support set
                       Q: Num of instances for each class in the query set
                       N_for_train：Num of classes for each batch for training
                       '''
            scheduler.step()  # 更新学习率
            support, query, label, support_n, query_n, label_n = self.train_data_loader.next_batch(B, N_for_train, K, Q)
           # support, query, label = self.train_data_loader.next_batch(B, N_for_train, K, Q)
            #support_n, query_n, label_n = self.train_data_loader.next_batch(B, N_for_train, K, Q)
            #logits, pred, dist = model(support_n, query_n, N_for_train, K, Q)
            logits, dist = model(support, query, N_for_train, K, Q)
            #loss = model.loss(logits, label)
            #logits_n, pred_n, dist_n = model(support_n, query_n, N_for_train, K, Q)
            logits_n, dist_n = model(support_n, query_n, N_for_train, K, Q)
            #loss_n = model.loss(logits_n, label_n)
            logits_1 = logits + logits_n
            loss = model.loss(logits_1, label)
            '''logits_n, pred_n, dist_n = model(support_n, query_n, N_for_train, K, Q)
            loss_n = model.loss(logits_n, label)'''

            #allloss = loss + dist + loss_n + dist_n
            allloss = loss + dist +dist_n
            #allloss = loss_n + dist_n
            optimizer.zero_grad()
            allloss.backward()
            optimizer.step()

            if (it + 1) % val_step == 0:
                with torch.no_grad():
                    acc = self.eval(model, 5, N_for_eval, K, 5, val_iter)
                    print(
                        "{0:}---{1:}-way-{2:}-shot test   Test accuracy: {3:3.2f}".format(it, N_for_eval, K, acc * 100))

                    if acc > best_acc:
                        print('Best checkpoint')
                        if not os.path.exists(ckpt_dir):
                            os.makedirs(ckpt_dir)
                        save_path = os.path.join(ckpt_dir, model_name + ".pth.tar")
                        torch.save({'state_dict': model.state_dict()}, save_path)
                        best_acc = acc
                model.train()

        print("\n####################\n")
        print("Finish training " + model_name)
        with torch.no_grad():
            test_acc = self.eval(model, 5, N_for_eval, K, 5, test_iter, ckpt=os.path.join(ckpt_dir, model_name + '.pth.tar'))

            print("{0:}-way-{1:}-shot test   Test accuracy: {2:3.2f}".format(N_for_eval, K, test_acc * 100))
            endtime = datetime.datetime.now()
            print((endtime - starttime).seconds)

    def eval(self, model, B, N, K, Q, eval_iter, ckpt=None):
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")
        if ckpt is None:
            eval_dataset = self.val_data_loader
        else:
            checkpoint = self.__load_model__(ckpt)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            eval_dataset = self.test_data_loader
        model.eval()

        iter_right = 0.0
        iter_sample = 0.0
        for it in range(eval_iter):
            #support, query, label, _, _, _ = eval_dataset.next_batch(B, N, K, Q)
            support, query, label, support_n, query_n, label_n = eval_dataset.next_batch(B, N, K, Q)
            logits, _ = model(support, query, N, K, Q)
            logits_n, _ = model(support_n, query_n, N, K, Q)
            _, pred = torch.max(logits+logits_n, 1)
            right = model.accuracy(pred, label)
            # .item()作用：取出单元素张量的元素值并返回该值，保持原元素类型不变。
            iter_right += right.item()
            iter_sample += 1
        return iter_right / iter_sample