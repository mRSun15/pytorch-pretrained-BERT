import sys, os, glob, random
import time
import parser
import torch
import torch.nn as nn
# from AdaAdam import AdaAdam
import torch.optim as OPT

from torchtext import data
import DataProcessing
from DataProcessing.MLTField import MTLField

from DataProcessing.NlcDatasetSingleFile import NlcDatasetSingleFile

batch_size = 10
seed = 12345678
gpu = 1
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.set_device(gpu)
    torch.cuda.manual_seed(seed)


def load_train_test_files(listfilename, test_suffix='.test'):
    filein = open(listfilename, 'r')
    file_tuples = []
    task_classes = ['.t2', '.t4', '.t5']
    for line in filein:
        array = line.strip().split('\t')
        line = array[0]
        for t_class in task_classes:
            trainfile = line + t_class + '.train'
            devfile = line + t_class + '.dev'
            testfile = line + t_class + test_suffix
            file_tuples.append((trainfile, devfile, testfile))
    filein.close()
    return file_tuples

filelist = 'data/Amazon_few_shot/workspace.filtered.list'
workingdir = 'data/Amazon_few_shot'
emfilename = 'glove.6B.300d'
emfiledir = '..'

datasets = []
list_datasets = []

file_tuples = load_train_test_files(filelist)
print(file_tuples)
TEXT = MTLField(lower=True)

for (trainfile, devfile, testfile) in file_tuples:
    print(trainfile, devfile, testfile)
    LABEL1 = data.Field(sequential=False)
    train1, dev1, test1 = NlcDatasetSingleFile.splits(
        TEXT, LABEL1, path=workingdir, train=trainfile,
        validation=devfile, test=testfile)
    datasets.append((TEXT, LABEL1, train1, dev1, test1))
    list_datasets.append(train1)
    list_datasets.append(dev1)
    list_datasets.append(test1)

datasets_iters = []

for (TEXT, LABEL, train, dev, test) in datasets:
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_size=batch_size, device=gpu)
    train_iter.repeat = False
    datasets_iters.append((train_iter, dev_iter, test_iter))

num_batch_total = 0
for i, (TEXT, LABEL, train, dev, test) in enumerate(datasets):
    print('DATASET%d'%(i+1))
    print('train.fields', train.fields)
    print('len(train)', len(train))
    print('len(dev)', len(dev))
    print('len(test)', len(test))
    print('vars(train[0])', vars(train[0]))
    num_batch_total += len(train) / batch_size

TEXT.build_vocab(list_datasets, vectors=emfilename,  vectors_cache=emfiledir)
# TEXT.build_vocab(list_dataset)

# build the vocabulary
for taskid, (TEXT, LABEL, train, dev, test) in enumerate(datasets):
    LABEL.build_vocab(train, dev, test)
    LABEL.vocab.itos = LABEL.vocab.itos[1:]
    for k, v in LABEL.vocab.stoi.items():
        LABEL.vocab.stoi[k] = v - 1

    # print vocab information
    print('len(TEXT.vocab)', len(TEXT.vocab))
    # print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

    #print LABEL.vocab.itos
    print(len(LABEL.vocab.itos))
    #if taskid == 0:
    #    print LABEL.vocab.stoi
    #print len(LABEL.vocab.stoi)
