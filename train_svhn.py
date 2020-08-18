import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
#from model_svhn import Network
from model import Network
from torch.utils.data import Dataset


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--rnn_cell', type=str, default='vanilla', help='type of RNN cell: vanilla, sigmoid, directional.')
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.025, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.7, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=25, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=3, help='num of init channels')
parser.add_argument('--layers', type=int, default=3, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=12, help='cutout length')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS_Vanilla', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=1, help='gradient clipping')
parser.add_argument('--nhid', type=int, default=256,
                    help='number of hidden units per layer')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10


def print_param_size(model):
  import numpy as np
  model_parameters = filter(
    lambda p: p.requires_grad, model.parameters()
  )
  params = sum([
    np.prod(p.size()) for p in model_parameters
  ])
  return params

class MyDataset(Dataset):
  def __init__(self, train, extra, transform):
    self.train = train
    self.extra = extra
    self.transform = transform
    self.train_len = len(train)
    self.extra_len = len(extra)

  def __getitem__(self, index):
    if index < self.train_len:
      x = self.train[index]
    else:
      index = index - self.train_len
      x = self.extra[index]
    return self.transform(x)

  def __len__(self):
    return self.train_len + self.extra_len


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.rnn_cell, args.init_channels, CIFAR_CLASSES, args.layers, args.nhid, genotype, ds="svhn")
  model = model.cuda()
  print("params:", print_param_size(model))

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss().cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  train_transform, valid_transform = utils._data_transforms_svhn_train(args)
  train_d = dset.SVHN(root=args.data, split="train", download=True, transform=train_transform)
  extra_d = dset.SVHN(root=args.data, split="extra", download=True, transform=train_transform)
  train_data = torch.utils.data.ConcatDataset([train_d, extra_d])

  valid_data = dset.SVHN(root=args.data, split="test", download=True, transform=valid_transform)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(args.epochs))

  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights_train.pt'))


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda(async=True)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)

    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()

