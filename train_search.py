import argparse
import torch
import time
import glob
import logging
import numpy as np
import sys
import os
import torchvision.datasets as dset
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn

from model_search import Network
from architect import Architect
import utils

logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser("cifar")
# seed params
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--save', type=str, default='EXP',
                    help='path to save the final model')

# dataset params
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')

# train params
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.7, help='momentum')
parser.add_argument('--clip', type=float, default=1,
                    help='gradient clipping')

# arch params
parser.add_argument('--rnn_cell', type=str, default='directional', help='type of RNN cell: vanilla, sigmoid, directional.')
parser.add_argument('--arch_learning_rate', type=float, default=3e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')

# model params
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--init_channels', type=int, default=3, help='num of init channels')
parser.add_argument('--layers', type=int, default=3, help='total number of layers')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

CIFAR_CLASSES = 10


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    # Setting seeds and CUDNN benchmark
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = False
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.rnn_cell, args.init_channels, CIFAR_CLASSES, args.layers, criterion, args.nhid)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)
        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        # training
        model, train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer)
        logging.info('train_acc %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        n = input.size(0)

        input = Variable(input).cuda()
        target = Variable(target).cuda()

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = Variable(input_search).cuda()
        target_search = Variable(target_search).cuda()

        architect.step(input, target, input_search, target_search, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))

        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            genotype = model.genotype()
            logging.info('genotype = %s', genotype)

    return model, top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda()

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
