

#import sys
# sys.path.append(r"../")
# C:\\Users\\Mohd Zia\\Desktop\\sem5\\mini_project\\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\\
#sys.path.append('C:\\Users\\Mohd Zia\\Desktop\\sem5\\mini_project\\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\\')
import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy
import torch.utils.data as data
#from data_iterator import dataIterator
#from Attention_RNN import AttnDecoderRNN
#from Densenet_torchvision import densenet121
from PIL import Image
from numpy import *

import pickle as pkl
import sys

import re
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict



from pathlib import Path

# Get the current working directory
current_directory = Path.cwd()


gpu=[0]
dictionaries=[current_directory/'dictionary.txt']
hidden_size = 256
batch_size_t = 1
maxlen = 100

def load_dict(dictFile):
    fp=open(dictFile)
    stuff=fp.readlines()
    fp.close()
    lexicon={}
    for l in stuff:
        w=l.strip().split()
        lexicon[w[0]]=int(w[1])
    print('total words/phones',len(lexicon))
    return lexicon

worddicts = load_dict(dictionaries[0])
worddicts_r = [None] * len(worddicts)
for kk, vv in worddicts.items():
	worddicts_r[vv] = kk


def for_test(x_t):

	h_mask_t = []
	w_mask_t = []
	encoder = densenet121()
	attn_decoder1 = AttnDecoderRNN(hidden_size,112,dropout_p=0.5)

	encoder = torch.nn.DataParallel(encoder, device_ids=gpu)
	attn_decoder1 = torch.nn.DataParallel(attn_decoder1, device_ids=gpu)
	encoder = encoder.cuda()
	attn_decoder1 = attn_decoder1.cuda()

	encoder.load_state_dict(torch.load(current_directory/'model/encoder_lr0.00001_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl'))
	attn_decoder1.load_state_dict(torch.load(current_directory/'model/attn_decoder_lr0.00001_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl'))

	encoder.eval()
	attn_decoder1.eval()

	x_t = Variable(x_t.cuda())
	x_mask = torch.ones(x_t.size()[0],x_t.size()[1],x_t.size()[2],x_t.size()[3]).cuda()
	x_t = torch.cat((x_t,x_mask),dim=1)
	x_real_high = x_t.size()[2]
	x_real_width = x_t.size()[3]
	h_mask_t.append(int(x_real_high))
	w_mask_t.append(int(x_real_width))
	x_real = x_t[0][0].view(x_real_high,x_real_width)
	output_highfeature_t = encoder(x_t)

	x_mean_t = torch.mean(output_highfeature_t)
	x_mean_t = float(x_mean_t)
	output_area_t1 = output_highfeature_t.size()
	output_area_t = output_area_t1[3]
	dense_input = output_area_t1[2]

	decoder_input_t = torch.LongTensor([111]*batch_size_t)
	decoder_input_t = decoder_input_t.cuda()

	decoder_hidden_t = torch.randn(batch_size_t, 1, hidden_size).cuda()
	# nn.init.xavier_uniform_(decoder_hidden_t)
	decoder_hidden_t = decoder_hidden_t * x_mean_t
	decoder_hidden_t = torch.tanh(decoder_hidden_t)

	prediction = torch.zeros(batch_size_t,maxlen)
	#label = torch.zeros(batch_size_t,maxlen)
	prediction_sub = []
	label_sub = []
	decoder_attention_t = torch.zeros(batch_size_t,1,dense_input,output_area_t).cuda()
	attention_sum_t = torch.zeros(batch_size_t,1,dense_input,output_area_t).cuda()
	decoder_attention_t_cat = []


	for i in range(maxlen):
		decoder_output, decoder_hidden_t, decoder_attention_t, attention_sum_t = attn_decoder1(
			decoder_input_t,
			decoder_hidden_t,
			output_highfeature_t,
			output_area_t,
			attention_sum_t,
			decoder_attention_t,
			dense_input,
			batch_size_t,
			h_mask_t,
			w_mask_t,
			gpu
    	)
		decoder_attention_t_cat.append(decoder_attention_t[0].data.cpu().numpy())
		topv,topi = torch.max(decoder_output,2)
		if torch.sum(topi)==0:
			continue
		decoder_input_t = topi
		decoder_input_t = decoder_input_t.view(batch_size_t)

	# prediction
	prediction[:,i] = decoder_input_t


	k = numpy.array(decoder_attention_t_cat)
	x_real = numpy.array(x_real.cpu().data)
	# print("before",prediction)
	prediction = prediction[0]
	# print("after",prediction)
	prediction_real = []
	for ir in range(len(prediction)):
		if int(prediction[ir]) == 0:
			continue
		prediction_real.append(worddicts_r[int(prediction[ir])])
	prediction_real.append('<eol>')
	print("prediction_real: ",prediction_real)
	# print("k:",k)

	prediction_real_show = numpy.array(prediction_real)


	return k,prediction_real_show

'''
Python 3.6
Pytorch >= 0.4
Written by Hongyu Wang in Beihang university
'''


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1,):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(self.dropout_p)

        self.embedding = nn.Embedding(self.output_size, 256)
        #self.gru = nn.GRUCell(684, 256)
        self.gru = nn.GRUCell(1024, self.hidden_size)
        self.gru1 = nn.GRUCell(256, self.hidden_size)
        self.out = nn.Linear(128, self.output_size)
        self.hidden = nn.Linear(self.hidden_size, 256)
        self.emb = nn.Linear(256, 128)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv_et = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
        self.conv_tan = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.hidden2 = nn.Linear(self.hidden_size, 128)
        self.emb2 = nn.Linear(256, 128)
        self.ua = nn.Linear(1024, 256)
        self.uf = nn.Linear(1, 256)
        self.v = nn.Linear(256, 1)
        self.wc = nn.Linear(1024, 128)
        self.bn = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, input_a, hidden, encoder_outputs,bb,attention_sum,decoder_attention,dense_input,batch_size,h_mask,w_mask,gpu):

        # batch_gpu must be an int object
        batch_gpu = int(batch_size/len(gpu))
        et_mask = torch.zeros(batch_gpu,dense_input,bb).cuda()

        if et_mask.device == torch.device('cuda:0'):
            for i in range(batch_gpu):
                et_mask[i][:h_mask[i],:w_mask[i]]=1

        if et_mask.device == torch.device('cuda:1'):
            for i in range(batch_gpu):
                et_mask[i][:h_mask[i+1*batch_gpu],:w_mask[i+1*batch_gpu]]=1

        if et_mask.device == torch.device('cuda:2'):
            for i in range(batch_gpu):
                et_mask[i][:h_mask[i+2*batch_gpu],:w_mask[i+2*batch_gpu]]=1

        if et_mask.device == torch.device('cuda:3'):
            for i in range(batch_gpu):
                et_mask[i][:h_mask[i+3*batch_gpu],:w_mask[i+3*batch_gpu]]=1

        et_mask_4 = et_mask.unsqueeze(1)

        # embedding the word from 1 to 256(total 112 words)
        embedded = self.embedding(input_a).view(batch_gpu,256)
        embedded = self.dropout(embedded)
        hidden = hidden.view(batch_gpu,self.hidden_size)

        st = self.gru1(embedded,hidden)
        hidden1 = self.hidden(st)
        hidden1 = hidden1.view(batch_gpu,1,1,256)

        # encoder_outputs from (batch,1024,height,width) => (batch,height,width,1024)
        encoder_outputs_trans = torch.transpose(encoder_outputs,1,2)
        encoder_outputs_trans = torch.transpose(encoder_outputs_trans,2,3)

        # encoder_outputs_trans (batch,height,width,1024) attention_sum_trans (batch,height,width,1) hidden1 (batch,1,1,256)
        decoder_attention = self.conv1(decoder_attention)
        attention_sum = attention_sum + decoder_attention
        attention_sum_trans = torch.transpose(attention_sum,1,2)
        attention_sum_trans = torch.transpose(attention_sum_trans,2,3)

        # encoder_outputs1 (batch,height,width,256) attention_sum1 (batch,height,width,256)
        encoder_outputs1 = self.ua(encoder_outputs_trans)
        attention_sum1 = self.uf(attention_sum_trans)

        et = hidden1 + encoder_outputs1 + attention_sum1
        et_trans = torch.transpose(et,2,3)
        et_trans = torch.transpose(et_trans,1,2)
        et_trans = self.conv_tan(et_trans)
        et_trans = et_trans*et_mask_4
        et_trans = self.bn1(et_trans)
        et_trans = torch.tanh(et_trans)
        et_trans = torch.transpose(et_trans,1,2)
        et_trans = torch.transpose(et_trans,2,3)

        et = self.v(et_trans) #4,9,34,1
        et = et.squeeze(3)
        # et = torch.transpose(et,2,3)
        # et = torch.transpose(et,1,2)
        # et = self.conv_et(et)
        # et = et*et_mask_4
        # et = self.bn(et)
        # et = self.relu(et)
        # et = et.squeeze(1)

        # et_div_all is attention alpha
        et_div_all = torch.zeros(batch_gpu,1,dense_input,bb)
        et_div_all = et_div_all.cuda()

        et_exp = torch.exp(et)
        et_exp = et_exp*et_mask
        et_sum = torch.sum(et_exp,dim=1)
        et_sum = torch.sum(et_sum,dim=1)
        for i in range(batch_gpu):
            et_div = et_exp[i]/(et_sum[i]+1e-8)
            et_div = et_div.unsqueeze(0)
            et_div_all[i] = et_div

        # ct is context vector (batch,128)
        ct = et_div_all*encoder_outputs
        ct = ct.sum(dim=2)
        ct = ct.sum(dim=2)

        # the next hidden after gru
        # batch,hidden_size
        hidden_next_a = self.gru(ct,st)
        hidden_next = hidden_next_a.view(batch_gpu, 1, self.hidden_size)

        # compute the output (batch,128)
        hidden2 = self.hidden2(hidden_next_a)
        embedded2 = self.emb2(embedded)
        ct2 = self.wc(ct)

        #output
        # output = F.log_softmax(self.out(hidden2+embedded2+ct2), dim=1)
        output = F.log_softmax(self.out(self.dropout(hidden2+embedded2+ct2)), dim=1)
        output = output.unsqueeze(1)

        return output, hidden_next, et_div_all, attention_sum

    def initHidden(self,batch_size):
        result = Variable(torch.randn(batch_size, 1, self.hidden_size))
        return result.cuda()

'''
Python 3.6
Pytorch >= 0.4
Written by Hongyu Wang in Beihang university
'''



def dataIterator(feature_file,label_file,dictionary,batch_size,batch_Imagesize,maxlen,maxImagesize):

    fp=open(feature_file,'rb')
    features=pkl.load(fp)
    fp.close()

    fp2=open(label_file,'r')
    labels=fp2.readlines()
    fp2.close()
    len_label = len(labels)

    targets={}
    # map word to int with dictionary
    for l in labels:
        tmp=l.strip().split()
        uid=tmp[0]
        w_list=[]
        for w in tmp[1:]:
            if w in dictionary:
                w_list.append(dictionary[w])
            else:
                print('a word not in the dictionary !! sentence ',uid,'word ', w)
                sys.exit()
        targets[uid]=w_list


    imageSize={}
    imagehigh={}
    imagewidth={}
    for uid,fea in features.items():
        imageSize[uid]=fea.shape[1]*fea.shape[2]
        imagehigh[uid]=fea.shape[1]
        imagewidth[uid]=fea.shape[2]

    imageSize= sorted(imageSize.items(), key=lambda d:d[1],reverse=True) # sorted by sentence length,  return a list with each triple element


    feature_batch=[]
    label_batch=[]
    feature_total=[]
    label_total=[]
    uidList=[]

    batch_image_size=0
    biggest_image_size=0
    i=0
    for uid,size in imageSize:
        if size>biggest_image_size:
            biggest_image_size=size
        fea=features[uid]
        lab=targets[uid]
        batch_image_size=biggest_image_size*(i+1)

        if len(lab)>maxlen:
            continue
            # print('sentence', uid, 'length bigger than', maxlen, 'ignore')

        elif size>maxImagesize:
            continue
            # print('image', uid, 'size bigger than', maxImagesize, 'ignore')

        else:
            uidList.append(uid)
            if batch_image_size>batch_Imagesize or i==batch_size: # a batch is full

                if label_batch:
                    feature_total.append(feature_batch)
                    label_total.append(label_batch)

                i=0
                biggest_image_size=size
                feature_batch=[]
                label_batch=[]
                feature_batch.append(fea)
                label_batch.append(lab)
                batch_image_size=biggest_image_size*(i+1)
                i+=1
            else:
                feature_batch.append(fea)
                label_batch.append(lab)
                i+=1

    # last
    feature_total.append(feature_batch)
    label_total.append(label_batch)
    len_ignore = len_label - len(feature_total)
    print('total ',len(feature_total), 'batch data loaded')
    print('ignore',len_ignore,'images')


    return feature_total,label_total

'''
Python 3.6
Pytorch >= 0.4
Written by Hongyu Wang in Beihang university
'''


__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

GroupNorm_num = 32
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        # self.add_module('norm1_G', nn.GroupNorm(GroupNorm_num,num_input_features)),
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        # self.add_module('norm2_G', nn.GroupNorm(GroupNorm_num,bn_size * growth_rate)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        # self.add_module('norm_G', nn.GroupNorm(GroupNorm_num,num_input_features))
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0_m', nn.Conv2d(2, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            # ('norm0_G', nn.GroupNorm(GroupNorm_num,num_init_features)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        # self.features.add_module('norm5_G', nn.GroupNorm(GroupNorm_num,num_features))
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.1
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        #out = F.avg_pool2d(out, kernel_size=2)
        # out = self.classifier(out)
        return out


def densenet121(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6,12,24),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model

