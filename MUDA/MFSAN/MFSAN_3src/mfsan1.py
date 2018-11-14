from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import math
import data_loader
import ResNet as models
from torch.utils import model_zoo
import numpy as np
import mmd
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--iter', type=int, default=15000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=8, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--l2_decay', type=float, default=5e-4,
                    help='the L2  weight decay')
parser.add_argument('--save_path', type=str, default="./tmp/origin_",
                    help='the path to save the model')
parser.add_argument('--root_path', type=str, default="/data/zhuyc/OfficeHome/",
                    help='the path to load the data')
parser.add_argument('--source1_dir', type=str, default="Art",#Art  Clipart   Product   Real World
                    help='the name of the source dir')
parser.add_argument('--source2_dir', type=str, default="Clipart",
                    help='the name of the source dir')
parser.add_argument('--source3_dir', type=str, default="Real World",
                    help='the name of the source dir')
parser.add_argument('--test_dir', type=str, default="Product",
                    help='the name of the test dir')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

#torch.manual_seed(args.seed)
#if args.cuda:
#    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

source1_loader = data_loader.load_training(args.root_path, args.source1_dir, args.batch_size, kwargs)
source2_loader = data_loader.load_training(args.root_path, args.source2_dir, args.batch_size, kwargs)
source3_loader = data_loader.load_training(args.root_path, args.source3_dir, args.batch_size, kwargs)
target_train_loader = data_loader.load_training(args.root_path, args.test_dir, args.batch_size, kwargs)
target_test_loader = data_loader.load_testing(args.root_path, args.test_dir, args.batch_size, kwargs)

def load_pretrain(model):
    url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    pretrained_dict = model_zoo.load_url(url)
    model_dict = model.state_dict()
    for k, v in model_dict.items():
        if "sharedNet" in k:
            model_dict[k] = pretrained_dict[k[k.find(".") + 1:]]

    model.load_state_dict(model_dict)
    return model

def train(model):
    #最后的全连接层学习率为前面的10倍
    source1_iter = iter(source1_loader)
    source2_iter = iter(source2_loader)
    source3_iter = iter(source3_loader)
    target_iter = iter(target_train_loader)
    correct = 0

    for i in range(1, args.iter + 1):
        model.train()
        LEARNING_RATE = args.lr / math.pow((1 + 10 * (i - 1) / (args.iter)), 0.75)
        if (i - 1) % 100 == 0:
            print("learning rate：", LEARNING_RATE)
        optimizer = torch.optim.SGD([
            {'params': model.sharedNet.parameters()},
            {'params': model.cls_fc_son1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc_son2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc_son3.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnet1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnet2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnet3.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=args.momentum, weight_decay=args.l2_decay)

        try:
            source_data, source_label = source1_iter.next()
        except Exception as err:
            source1_iter = iter(source1_loader)
            source_data, source_label = source1_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        if args.cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        optimizer.zero_grad()

        cls_loss, mmd_loss, l1_loss = model(source_data, target_data, source_label, source_label, 1)
        gamma = 2 / (1 + math.exp(-10 * (i) / (args.iter) )) - 1
        loss = cls_loss + gamma * (mmd_loss)# + l1_loss)
        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                i, 100. * i / args.iter, loss.data[0], cls_loss.data[0], mmd_loss.data[0], l1_loss.data[0]))

        #if i % 3 == 2:
        try:
            source_data, source_label = source2_iter.next()
        except Exception as err:
            source2_iter = iter(source2_loader)
            source_data, source_label = source2_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        if args.cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        optimizer.zero_grad()

        cls_loss, mmd_loss, l1_loss = model(source_data, target_data, source_label, source_label, 2)
        gamma = 2 / (1 + math.exp(-10 * (i) / (args.iter))) - 1
        loss = cls_loss + gamma * (mmd_loss)# + l1_loss)
        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print(
                'Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                    i, 100. * i / args.iter, loss.data[0], cls_loss.data[0], mmd_loss.data[0], l1_loss.data[0]))

        #source3
        try:
            source_data, source_label = source3_iter.next()
        except Exception as err:
            source3_iter = iter(source3_loader)
            source_data, source_label = source3_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        if args.cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        optimizer.zero_grad()

        cls_loss, mmd_loss, l1_loss = model(source_data, target_data, source_label, source_label, 3)
        gamma = 2 / (1 + math.exp(-10 * (i) / (args.iter) )) - 1
        loss = cls_loss + gamma * (mmd_loss)# + l1_loss)
        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('Train source3 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                i, 100. * i / args.iter, loss.data[0], cls_loss.data[0], mmd_loss.data[0], l1_loss.data[0]))

        if i == args.iter:
            print("aaaaaa")
        if i % (args.log_interval * 20) == 0:
            t_correct = test(model)
            if t_correct > correct:
                correct = t_correct
            print(args.source1_dir, args.source2_dir, "to", args.test_dir, "%s max correct:" % args.test_dir, correct, "\n")

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0

    for data, target in target_test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        pred1, pred2, pred3 = model(data)

        #print(weight)
        test_loss += F.nll_loss(F.log_softmax(pred1, dim = 1), target, size_average=False).data[0] # sum up batch loss
        pred1 = torch.nn.functional.softmax(pred1, dim=1)
        pred2 = torch.nn.functional.softmax(pred2, dim=1)
        pred3 = torch.nn.functional.softmax(pred3, dim=1)

        pred = (pred1 + pred2 + pred3) / 3
        pred = pred.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        pred = pred1.data.max(1)[1]  # get the index of the max log-probability
        correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
        pred = pred2.data.max(1)[1]  # get the index of the max log-probability
        correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()
        pred = pred3.data.max(1)[1]  # get the index of the max log-probability
        correct3 += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(target_test_loader.dataset)
    print(args.test_dir, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(target_test_loader.dataset),
        100. * correct / len(target_test_loader.dataset)))
    print(correct1, correct2, correct3)
    return correct

if __name__ == '__main__':
    model = models.DANNet(num_classes=65)
    print(model)
    if args.cuda:
        model.cuda()
    #model = load_pretrain(model)
    train(model)
    #test(model)
