# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 08:17:59 2020
Build a neural network which can classify the watermelon
@author: 未海
"""
import numpy as np
class Network(object):
    #神经网络的初始化，获取网络层数，初始化权重和偏差
    def __init__(self,sizes):
        self.layers=len(sizes)
        self.sizes=sizes
        self.bias=[np.random.randn(y,1) for y in sizes[1:] ]
        self.weights=[np.random.randn(y,x)for x,y in zip(sizes[:-1],sizes[1:])]
    #前向传播，计算最终输出值
    def feedword(self,a):
        for w,b in zip(self.weights,self.bias):
            a=sigmoid(np.dot(w,a)+b)
        return a
    #反向传播的计算
    def backward(self,x,y):
        #生成db和dw两个列表储存b和w的变化，为更新做准备
        db = [np.zeros(b.shape) for b in self.bias]
        dw = [np.zeros(w.shape) for w in self.weights]
        #存储每层神经元经过激活函数后的值，第一列存储为输入值
        activation=x
        activations=[x]
        #以列表zs存储未经激活函数计算，输入到每层神经元的值
        zs=[]
        '''更新输出层与次输出层之间的权重矩阵时，需要输出层的输出与输出层的输入方可更新，更新
        中间层n与次层n-1之间的权重矩阵时，需要该层的输出与输入（与输出层类似），但同时，因为
        中间层有与上一层的连接，因此中间层的梯度受紧邻的上一层的影响，也只受这一层的影响，因此
        需要上一层的各个梯度和上一层的权重矩阵，故有如下计算过程'''
        #计算输出层与次输出层之间权重与阈值的更新矩阵
        for w,b in zip(self.weights,self.bias):
            z=np.dot(w,activation)+b
            zs.append(z)
            activation=sigmoid(z)
            activations.append(activation)
        g=(activations[-1]-y)*sigmoid_dot(zs[-1])
        '''这里的计算需要转置，矩阵g的每一行对应于一个输出神经元的梯度，因此需要乘上次输出层
        的输出的矩阵的转置，才可以成为一个权重更新矩阵.也就是说，该矩阵的每一行对应一个输出
        神经元，每一行中的每一个元素对应一个次输出层的神经元'''
        dw[-1]=np.dot(g,activations[-2].transpose())
        db[-1]=g
        #计算中间层间的权重与阈值更新矩阵
        for j in range(2,self.layers):
            z=zs[-j]
            g=np.dot(self.weights[-j+1].transpose(),g)*sigmoid_dot(z)
            dw[-j]=np.dot(g,activations[-j-1].transpose())
            db[-j]=g
        return db,dw
    #根据计算得到的更新矩阵更新权重矩阵和阈值矩阵       
    def update_w_b(self,mini_batch,learning_rate):
        # 根据 bias 和 weights 的行列数创建对应的全部元素值为 0 的空矩阵
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #根据小样本计算累计误差（实现随机梯度下降）
        for x, y in mini_batch:
            # 根据样本中的每一个输入 x 的其输出 y，计算 w 和 b 的偏导数
            delta_nabla_b, delta_nabla_w = self.backward(x, y)
            # 累加储存偏导值 delta_nabla_b 和 delta_nabla_w
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # 更新根据累加的偏导值更新 w 和 b，注意要除以小样本的样本数
        self.weights = [w-(learning_rate/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate/len(mini_batch))*nb
                       for b, nb in zip(self.bias, nabla_b)]
    #实现随机梯度下降
    def SGD(self,training_data,mini_batch_size,times,learning_rate,test_data):
        n_tra=len(training_data)
        n_test=len(test_data)
        #开始进行迭代
        for i in range(times):
            #生成用于随机梯度下降的小样本（在训练集中随机抽取即可）
            mini_batchs=[training_data[k:k+mini_batch_size]for k in range (0,n_tra,mini_batch_size)]
            #进行更新
            for mini_batch in mini_batchs:
                self.update_w_b(mini_batch,learning_rate)
            #输出训练模型的精度
            print("Epoh{0}:{1}/{2}".format(i,self.evaluate(test_data),n_test))
    #用于精度评估的函数
    def evaluate(self,test_data):
        #下面一行也有问题
        test_results=[(np.argmax(self.feedword(x)),y)for (x,y)in test_data]
        sum_right=sum(int(x==y)for (x,y)in test_results)
        return sum_right
#激活函数（为简便计，激活函数选择sigmoid函数）   
def sigmoid(a):   
    z=1/(1+np.exp(-a))
    return z 
#激活函数的导数
def sigmoid_dot(a):
    z=sigmoid(a)*(1-sigmoid(a))
    return z 

        
        

