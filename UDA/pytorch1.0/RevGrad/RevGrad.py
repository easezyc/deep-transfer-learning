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
import torchvision
import torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
batch_size = 32
iteration=10000
lr = 0.01
momentum = 0.9
no_cuda =False
seed = 8
log_interval = 10
l2_decay = 5e-4
root_path = "/data/zhuyc/OFFICE31/"
src_name = "dslr"
tgt_name = "amazon"

cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

src_loader = data_loader.load_training(root_path, src_name, batch_size, kwargs)
tgt_train_loader = data_loader.load_training(root_path, tgt_name, batch_size, kwargs)
tgt_test_loader = data_loader.load_testing(root_path, tgt_name, batch_size, kwargs)

src_dataset_len = len(src_loader.dataset)
tgt_dataset_len = len(tgt_test_loader.dataset)
src_loader_len = len(src_loader)
tgt_loader_len = len(tgt_train_loader)


def train(model):
    src_data_iter = iter(src_loader)
    tgt_data_iter = iter(tgt_train_loader)
    src_dlabel = Variable(torch.ones(batch_size).long().cuda())
    tgt_dlabel = Variable(torch.zeros(batch_size).long().cuda())
    correct=0
    #gradient_reverse_layer = network.AdversarialLayer(high_value=config["high"])
    for i in range(1, iteration+1):
        model.train()
        LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        if (i-1)%100==0:
            print("learning rate: ", LEARNING_RATE)
        optimizer_fea = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.cls_fn.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)
        optimizer_critic = torch.optim.SGD([
        {'params': model.domain_fn.parameters(), 'lr': LEARNING_RATE}
        ], lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)

        try:
            src_data, src_clabel = src_data_iter.next()
            tgt_data, tgt_label = tgt_data_iter.next()
        except Exception as err:
            src_data_iter = iter(src_loader)
            src_data, src_clabel = src_data_iter.next()
            tgt_data_iter = iter(tgt_train_loader)
            tgt_data, tgt_label = tgt_data_iter.next()
   
        if i % tgt_loader_len == 0:
            tgt_data_iter = iter(tgt_train_loader)  
        if cuda:
            src_data, src_clabel = src_data.cuda(), src_clabel.cuda()
            tgt_data, tgt_label = tgt_data.cuda(), tgt_label.cuda()
        src_clabel_pred, src_dlabel_pred = model(src_data)
        loss=nn.CrossEntropyLoss()
        label_loss=loss(src_clabel_pred,src_clabel)
            
        tgt_clabel_pred, tgt_dlabel_pred = model(tgt_data)
        new_label_pred=torch.cat((src_dlabel_pred,tgt_dlabel_pred),0)
        confusion_loss=nn.BCELoss()
        confusion_loss_total=confusion_loss(new_label_pred,torch.cat((src_dlabel,tgt_dlabel),0).float().reshape(2*batch_size,1))

        fea_loss_total = confusion_loss_total + label_loss
        optimizer_fea.zero_grad()
        fea_loss_total.backward()
        optimizer_fea.step()

        if i % log_interval == 0:
            print('Train iter: {} [({:.0f}%)]\tconfusion_Loss: {:.6f}\tlabel_Loss: {:.6f}'.format(
                i, 100. * i / iteration, confusion_loss_total.item(), label_loss.item()))

        if i%(log_interval*20)==0:
            t_correct = test(model)
            if t_correct > correct:
                correct = t_correct
            print('src: {} to tgt: {} max correct: {} max accuracy{: .2f}%\n'.format(
              src_name, tgt_name, correct, 100. * correct / tgt_dataset_len ))

       
def test(model):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for tgt_test_data, tgt_test_label in tgt_test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)
            tgt_clabel_pred, tgt_dlabel_pred = model(tgt_test_data)
            test_loss += F.nll_loss(F.log_softmax(tgt_clabel_pred, dim = 1), tgt_test_label, reduction='sum').item() # sum up batch loss
            pred = tgt_clabel_pred.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()

    test_loss /= tgt_dataset_len
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        tgt_name, test_loss, correct, tgt_dataset_len,
        100. * correct / tgt_dataset_len))
    return correct


if __name__ == '__main__':
    model = models.RevGrad(num_classes=31)
    print(model)
    if cuda:
        model.cuda()
    train(model)
