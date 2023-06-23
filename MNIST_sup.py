import torch
from torch.utils import data
import torch.optim as optim
import torchvision.transforms as transforms
import scipy
from sklearn.linear_model import LinearRegression
# from estimators import twonn
from sklearn.neighbors import NearestNeighbors
import torchvision.models as models
import matplotlib
import matplotlib.pyplot as plt

# plt.style.use('ggplot')
matplotlib.use('TkAgg')

# from ScaleSteerableInvariant_Network import *
# from ScaleSteerableInvariant_Network_groupeq import *
from Networks import *
# from VanillaCNN import *

import numpy as np
import sys, os
from utils import Dataset, load_dataset


import random
# #
# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# torch.backends.cudnn.deterministic=True



import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bsize', default=1024, type=int,
                        help='batch size for previous images')
    parser.add_argument('--n-workers', default=1, type=int)
    parser.add_argument('--anchor-samples', default=0, type=int,
                        help="0 for using all samples from the training set")
    parser.add_argument('--anchor-ratio', default=0, type=float,
                        help="0 for using all samples from the training set")
    parser.add_argument('--max_num_samples', default=-1, type=int,
                        help="Maximum number of samples to process." \
                             "Useful for evaluating convergence.")


    args, _ = parser.parse_known_args()
    return args


def estimate_id(X, plot=False, X_is_dist=False):
    # INPUT:
    #   X = Nxp matrix of N p-dimensional samples (when X_is_dist is False)
    #   plot = Boolean flag of whether to plot fit
    #   X_is_dist = Boolean flag of whether X is an NxN distance metric instead
    #
    # OUTPUT:
    #   d = TWO-NN estimate of intrinsic dimensionality

    N = X.shape[0]

    if X_is_dist:
        dist = X
    else:
        # COMPUTE PAIRWISE DISTANCES FOR EACH POINT IN THE DATASET
        dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X, metric='euclidean'))

    # FOR EACH POINT, COMPUTE mu_i = r_2 / r_1,
    # where r_1 and r_2 are first and second shortest distances
    mu = np.zeros(N)

    for i in range(N):
        sort_idx = np.argsort(dist[i,:])
        mu[i] = dist[i,sort_idx[2]] / dist[i,sort_idx[1]]


    # COMPUTE EMPIRICAL CUMULATE
    sort_idx = np.argsort(mu)
    Femp     = np.arange(N)/N

    # FIT (log(mu_i), -log(1-F(mu_i))) WITH A STRAIGHT LINE THROUGH ORIGIN
    lr = LinearRegression(fit_intercept=False)
    lr.fit(np.log(mu[sort_idx]).reshape(-1,1), -np.log(1-Femp).reshape(-1,1))

    d = lr.coef_[0][0] # extract slope

    if plot:
        # PLOT FIT THAT ESTIMATES INTRINSIC DIMENSION
        s=plt.scatter(np.log(mu[sort_idx]), -np.log(1-Femp), c='r', label='data')
        p=plt.plot(np.log(mu[sort_idx]), lr.predict(np.log(mu[sort_idx]).reshape(-1,1)), c='k', label='linear fit')
        plt.xlabel('$\log(\mu_i)$'); plt.ylabel('$-\log(1-F_{emp}(\mu_i))$')
        plt.title('ID = ' + str(np.round(d, 3)))
        plt.legend()

    return d

import skdim
import time
def id_entropy(feats,K_val,every_k):

    neigh = NearestNeighbors(n_neighbors=K_val)
    # feats_energy = np.repeat(np.expand_dims(np.sqrt(np.mean(feats**2,0)),0),feats.shape[0],0)
    # feats = feats/feats_energy
    # feats = np.transpose(feats)
    # feats = (feats - feats.mean(axis=0)) / feats.std(axis=0)
    # feats = np.transpose(feats)
    # print(feats.shape)
    neigh.fit(feats)
    id_all = []
    total = 0
    for i in range(0,feats.shape[0],every_k):
        data = neigh.kneighbors(feats[i].reshape(1,-1),return_distance=False)
        # t1 = time.time()
        # id_all.append(estimate_id(np.squeeze(feats[data]) ))
        # print('here')
        # print(feats[data].shape)
        fisherS = skdim.id.FisherS().fit(np.squeeze(feats[data]))
        # t2 = time.time()
        # print(t2 - t1)

        if not(np.isnan(fisherS.dimension_)):
            id_all.append(fisherS.dimension_)

    id_ent = np.mean(np.array(id_all))

    return id_ent



# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# This is the testbench for the
# MNIST-Scale, FMNIST-Scale and CIFAR-10-Scale datasets.
# The networks and network architecture are defiend
# within their respective libraries


def train_network(net, trainloader, init_rate, step_size, gamma, total_epochs, weight_decay, testloader):

    net = net
    net = net.cuda()
    net = net.train()
    # params = add_weight_decay(net, l2_normal,l2_special,name_special)
    optimizer = optim.SGD(net.parameters(), lr=init_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss()



    id_ents = [[],[],[],[]]
    accuracies = []
    train_accuracies = []
    for epoch in range(total_epochs):

        # print("Time for one epoch:",time.time()-s)
        # s = time.time()

        # torch.cuda.empty_cache()
        scheduler.step()
        # print('epoch: ' + str(epoch))
        feats_all = [[],[],[],[]]
        if np.mod(epoch, 3) == 0:
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs,feats = net(inputs)

                if len(feats_all[0]) == 0:
                    for i in range(3,4):
                        feats_all[i] = feats[i].detach()

                else:
                    for i in range(3,4):
                        feats_all[i] = torch.cat((feats_all[i],feats[i].detach()),0)

            for i in range(3,4):
                id_ents[i].append(id_entropy(torch.squeeze(feats_all[i]).detach().cpu().numpy(), 200, 50))

            # loss = criterion(outputs, labels)

            # loss.backward()
            # optimizer.step()

        # print('aaa')




    # else:
        #     for i in range(1,4):
        #         id_ents[i].append(id_entropy(torch.squeeze(feats_all[i]).detach().cpu().numpy(),100,100))

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs, feats = net(inputs)
            # if len(feats_all) == 0:
            #     feats_all = feats.detach()
            # else:
            #     feats_all = torch.cat((feats_all, feats.detach()), 0)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        # id_ents.append(estimate_id(torch.squeeze(feats_all).detach().cpu().numpy()))
        # print(id_ents[0][-1])
        # print(id_ents[1][-1])
        # print(id_ents[2][-1])
        # print(id_ents[3][-1])

        if np.mod(epoch, 3) == 0:
            accuracy = test_network(net, testloader, test_labels)
            accuracy2 = test_network(net, trainloader_small, train_labels)
            # print("Test:", accuracy)
            accuracies.append(accuracy)
            train_accuracies.append(accuracy2)
            # a = input('')
            net = net.train()

        # print('break')
    net = net.eval()

    # plt.plot(id_ents[0],label="0")
    # plt.plot(id_ents[1],label="1")
    # plt.plot(id_ents[2],label="2")
    # plt.plot(id_ents[3],label="3")
    # plt.plot(accuracies)
    # plt.plot(train_accuracies)
    # plt.legend()
    # plt.show()
    # plt.waitforbuttonpress()

    return id_ents, accuracies, train_accuracies, net


def train_network_fin_id(net, trainloader, init_rate, step_size, gamma, total_epochs, weight_decay, testloader):

    net = net
    net = net.cuda()
    net = net.train()
    # params = add_weight_decay(net, l2_normal,l2_special,name_special)
    optimizer = optim.SGD(net.parameters(), lr=init_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss()



    id_ents = []
    accuracies = []
    train_accuracies = []
    for epoch in range(total_epochs):

        # print("Time for one epoch:",time.time()-s)
        # s = time.time()

        # torch.cuda.empty_cache()

        # print('epoch: ' + str(epoch))
        feats_all = [[],[],[],[]]

        if epoch == total_epochs - 1:
            net = net.eval()
            for i, data in enumerate(testloader_small, 0):
                # get the inputs
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs,feats = net(inputs.float())

                if len(feats_all[0]) == 0:
                    for i in range(4):
                        feats_all[i] = feats[i].detach()

                else:
                    for i in range(4):
                        feats_all[i] = torch.cat((feats_all[i],feats[i].detach()),0)



            tempa = []
            for i in range(3,4):
                tempa.append(id_entropy(torch.squeeze(feats_all[i]).detach().cpu().numpy(),200,50))
            id_ents.append(tempa)
            # print(id_ents)

            net = net.train()

            # loss = criterion(outputs, labels)

            # loss.backward()
            # optimizer.step()

        # print('aaa')

        # if np.mod(epoch,3) == 0:
        #     for i in range(4):
        #         id_ents[i].append(id_entropy(torch.squeeze(feats_all[i]).detach().cpu().numpy(),20,100))

    # else:
        #     for i in range(1,4):
        #         id_ents[i].append(id_entropy(torch.squeeze(feats_all[i]).detach().cpu().numpy(),100,100))

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs, feats = net(inputs)
            # if len(feats_all) == 0:
            #     feats_all = feats.detach()
            # else:
            #     feats_all = torch.cat((feats_all, feats.detach()), 0)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        scheduler.step()
        # id_ents.append(estimate_id(torch.squeeze(feats_all).detach().cpu().numpy()))
        # print(id_ents[0][-1])
        # print(id_ents[1][-1])
        # print(id_ents[2][-1])
        # print(id_ents[3][-1])

        if epoch == total_epochs - 1:
            accuracy = test_network(net, testloader, test_labels)
            accuracy2 = test_network(net, trainloader_small, train_labels)
            # print("Test:", accuracy)
            # accuracies.append(50*accuracy)
            # train_accuracies.append(50 * accuracy2)
            # a = input('')
            net = net.train()

        # print('break')
    net = net.eval()
    id_ents = np.mean(np.array(id_ents),0)
    # print('hh')

    # plt.plot(id_ents[0],label="0")
    # plt.plot(id_ents[1],label="1")
    # plt.plot(id_ents[2],label="2")
    # plt.plot(id_ents[3],label="3")
    # plt.plot(accuracies)
    # plt.plot(train_accuracies)
    # plt.legend()
    # plt.show()
    # plt.waitforbuttonpress()

    return id_ents, accuracy, accuracy2, net


def test_network(net, testloader, test_labels):

    net = net.eval()

    correct = torch.tensor(0)
    total = len(test_labels)
    dataiter = iter(testloader)
    # print(len(test_labels))

    for i in range(int(len(test_labels) / testloader.batch_size)):
        images, labels = dataiter.next()
        images = images.cuda()
        labels = labels.cuda()
        outputs,feats = net(images.float())
        _, predicted = torch.max(outputs, 1)
        correct = correct + torch.sum(predicted == labels)
        # torch.cuda.empty_cache()

    accuracy = float(correct)/float(total)
    return accuracy



def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def noisy_labels(labels,noise_prob):
    max_labels = np.max(labels)
    rand_labels = np.random.randint(0,max_labels+1,len(labels))
    rand_arr = (np.random.rand(len(labels)) > 1 - noise_prob).astype(float)

    noisy_labels = (labels*(1-rand_arr)) + (rand_labels*(rand_arr))
    noisy_labels = noisy_labels.astype(int)
    return noisy_labels


def seed_worker(worker_id):
    bablu = 1
    # np.random.seed(0)
    # random.seed(0)
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)
    # torch.backends.cudnn.deterministic = True

from copy import deepcopy

if __name__ == "__main__":

    args = parse_args()

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    dataset_name = 'MNIST'
    val_splits = 1
    training_size = 50000
    training_sizes = [50000]
    batch_size = 400
    init_rate = 0.04
    decay_normal = 0.0001
    step_size = 10
    gamma = 0.7
    total_epochs = 50
    noisy_label_prob = 0
    num_workers = 0
    no_reruns = 20

    # writepath = './result/stl10_dct.txt'
    # mode = 'a+' if os.path.exists(writepath) else 'w+'
    # f = open(writepath, mode)
    # f.write('Number of epoch is: ' + str(total_epochs) + '\n')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #
    # augmentation = transforms.RandomApply([
    #     transforms.RandomResizedCrop(96, scale=(0.7, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(10)
    # ])z



        # normalize])

    # transform_train = transforms.Compose([transforms.ToTensor()])
    # transform_test = transforms.Compose([transforms.ToTensor()])

    # Networks_to_train = [SimpleCNN_Network(), Net_steerinvariant_stl10_scale(), Net_steergroupeq_stl10_scale(), Net_steergroupeq_stl10_scale_dctbasis()]

    # Networks_to_train = [Net_steergroupeq_stl10_RotoScale(ms_interact=0)]
    Networks_to_train = [0]
    # Networks_to_train = [Net_steerinvariant_mnist_scale(k_range=[0.5]),Net_steerinvariant_mnist_scale(k_range=[0.5,1]),
    #                      Net_steerinvariant_mnist_scale(k_range=[0.5,1,2]),Net_steerinvariant_mnist_scale(k_range=[0.5,1,2,3]),
    #                      Net_steerinvariant_mnist_scale(k_range=[0.5,1,2,3,4])]

    accuracy_all = np.zeros((val_splits, len(Networks_to_train)))

    for idx in range(len(training_sizes)):
        listdict = load_dataset(dataset_name, val_splits, training_sizes[idx])
        # f.write('%d test cycle: \n' % (idx + 1))

        for i in range(val_splits):
            Networks_to_train = [Net_vanilla_cnn_mnist()]

            if dataset_name == 'MNIST':
                transform_train = transforms.Compose([
                    # transforms.Lambda(lambda x: x.convert("RGB")),
                    # augmentation,
                    # transforms.ToPILImage(),
                    # transforms.RandomAffine(0,(0.1,0.1)),
                    # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    transforms.ToTensor(), ])
                # normalize])

                transform_test = transforms.Compose([
                    # transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(), ])
                # normalize])
                listdict = load_dataset(dataset_name, val_splits, training_sizes[0])
                train_data = listdict[i]['train_data']
                # train_data = dataset_transform(train_data,transform_type,transform_params)
                train_labels = listdict[i]['train_label']

                if noisy_label_prob > 0:
                    train_labels = noisy_labels(train_labels, noisy_label_prob)
                    print('here i am')

                test_data = listdict[i]['test_data']
                # test_data = dataset_transform(test_data,transform_type,transform_params)
                test_labels = listdict[i]['test_label']

                Data_train = Dataset(dataset_name, train_data, train_labels, transform_train)
                # Data_train = Dataset_NearestNeighbor(dataset_name, train_data, train_labels, transform_train)

                Data_test = Dataset(dataset_name, test_data, test_labels, transform_test)
                Data_test_small = Dataset(dataset_name, test_data[0:2000, :, :], test_labels[0:2000], transform_train)

                trainloader = torch.utils.data.DataLoader(Data_train, batch_size=batch_size, shuffle=False,
                                                          num_workers=0, worker_init_fn=seed_worker)
                trainloader_small = torch.utils.data.DataLoader(Data_train, batch_size=400, shuffle=False, num_workers=0,
                                                                worker_init_fn=seed_worker)

                testloader = torch.utils.data.DataLoader(Data_test, batch_size=200, shuffle=False, num_workers=0,
                                                         worker_init_fn=seed_worker)
                testloader_small = torch.utils.data.DataLoader(Data_test_small, batch_size=batch_size, shuffle=False, num_workers=0,
                                                         worker_init_fn=seed_worker)

            if dataset_name == 'CIFAR10':
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                transform_train = transforms.Compose(
                    [transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize, ])

                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])

                trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                        download=True, transform=transform_train)
                if noisy_label_prob>0:
                    trainset.targets = (noisy_labels(trainset.targets, noisy_label_prob)).astype('long')

                trainset.data = trainset.data[0:training_size, :, :, :]
                trainset.targets = trainset.targets[0:training_size]

                trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                          shuffle=False, num_workers=num_workers,multiprocessing_context='spawn' )

                testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                       download=True, transform=transform_test)

                # testset.data = testset.data[0:test_size, :, :, :]
                # testset.targets = testset.targets[0:test_size]

                test_labels = np.array(testset.targets)
                testloader = torch.utils.data.DataLoader(testset, batch_size=20,
                                                         shuffle=False, num_workers=num_workers,multiprocessing_context='spawn' )

            # net1 = torch.load('./dct_eqv_3scale_stl10_5k_training'+str(i)+'.pt')
            # net2 = torch.load('simple_cnn_stl10_5k_training0.pt')

            print('im here')

            for j in range(len(Networks_to_train)):

                if no_reruns == 1:
                    net = train_network(Networks_to_train[0], trainloader, init_rate, step_size, gamma, total_epochs, decay_normal, testloader)
                else:
                    id_ents_all = []
                    accuracy_all = []
                    accuracy2_all = []
                    # decay_normal = [0.01,0.02,0.05,0.09,0.001,0.002,0.005,0.0001,0.0005,0]
                    # noisy_probs = [0,0,0,0,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.2,0.3,0.3,0.3,0.3,0.4,0.4,0.4,0.4,0.5,0.5,0.5,0.5]
                    # data_num = [200,300,400,500, 600, 700, 800, 900, 1000,2500,5000,7500,20000,60000]
                    data_num = [5000,7500, 10000, 12500, 15000, 17500,20000,25000, 30000, 35000]

                    # data_num = [60000]
                    for temp in range(len(data_num)):
                        # train_labels = listdict[i]['train_label']
                        Data_train2 = Dataset(dataset_name, train_data[0: data_num[temp], :, :], train_labels[0:data_num[temp]], transform_train)
                        # train_labels_new = noisy_labels(deepcopy(train_labels), noisy_probs[temp])
                        # Data_train2 = Dataset(dataset_name, train_data, train_labels_new, transform_train)
                        trainloader = torch.utils.data.DataLoader(Data_train2, batch_size=batch_size, shuffle=False,
                                                                  num_workers=0, worker_init_fn=seed_worker)

                        id_ents_temp = []
                        accuracy_temp = []
                        accuracy2_temp = []

                        for tempa in range(8):
                            id_ents, accuracy, accuracy2, net = train_network_fin_id(Net_vanilla_cnn_mnist(), trainloader, init_rate, step_size, gamma,
                                                total_epochs, decay_normal, testloader)
                            # print(id_ents)
                            id_ents_temp.append(id_ents)
                            accuracy_temp.append(accuracy)
                            accuracy2_temp.append(accuracy2)
                        #
                        # net = 0
                        id_ents = np.mean(np.array(id_ents_temp))

                        accuracy = np.mean(np.array(accuracy_temp))
                        accuracy2 = np.mean(np.array(id_ents_temp))
                        #
                        print('--------------------------Average ID-Entropy---------------------------------')
                        print(id_ents)
                        print('--------------------------Average Test-Entropy---------------------------------')
                        print(accuracy)
                        id_ents_all.append(id_ents)
                        accuracy_all.append(accuracy)
                        accuracy2_all.append(accuracy2)

                        # id_ents, accuracies, train_accuracies, net = train_network(deepcopy(Networks_to_train[0]), trainloader, init_rate, step_size, gamma,
                        #                     total_epochs, decay_normal, testloader)
                        # net = 0
                        # print("ID-Entropy after every 3 epochs:")
                        # print(id_ents[3])
                        # print("Test Accuracy after every 3 epochs:")
                        # print(accuracies)
                        # id_ents_all.append(id_ents[3])
                        # accuracy_all.append(accuracies)
                        # accuracy2_all.append(train_accuracies)

                # fig, ax1 = plt.subplots()
                # ax1.set_xlabel('Number of Training Epochs')
                # ax1.set_ylabel('ID-Entropy')
                # ax1.set_xlabel('Number of Training Epochs', fontsize=30)
                # ax1.set_ylabel('ID-Entropy', fontsize=30)
                # ax2 = ax1.twinx()
                # ax2.set_ylabel('Test Accuracy', fontsize=30)
                # ax1.plot(id0, label='ID-Entropy (p=0)')
                # ax1.fill_between(list(range(17)), id0 - (id0_std / 2), id0 + (id0_std / 2), alpha=0.2)
                # ax2.plot(acc0, '--', label='Test Accuracy (p=0)')
                # ax2.fill_between(list(range(17)), acc0 - (acc0_std / 2), acc0 + (acc0_std / 2), alpha=0.2)
                # ax1.plot(id1, label='ID-Entropy (p=0.1)')
                # ax1.fill_between(list(range(17)), id1 - (id1_std / 2), id1 + (id1_std / 2), alpha=0.2)
                # ax2.plot(acc1, '--', label='Test Accuracy (p=0.1)')
                # ax2.fill_between(list(range(17)), acc1 - (acc1_std / 2), acc1 + (acc1_std / 2), alpha=0.2)
                # ax1.plot(id2, label='ID-Entropy (p=0.2)')
                # ax1.fill_between(list(range(17)), id2 - (id2_std / 2), id2 + (id2_std / 2), alpha=0.2)
                # ax2.plot(acc2, '--', label='Test Accuracy (p=0.2)')
                # ax2.fill_between(list(range(17)), acc2 - (acc2_std / 2), acc2 + (acc2_std / 2), alpha=0.2)
                # ax1.tick_params(axis='both', which='major', labelsize=20)
                # ax2.tick_params(axis='both', which='major', labelsize=20)
                # ax1.legend()
                # ax2.legend()
                # plt.show()
                # ax1.legend(fontsize=15)
                # ax2.legend(fontsize=15)
               #  id0 = (id_ents_all_arr[0][3] + id_ents_all_arr[1][3] + id_ents_all_arr[2][3] + id_ents_all_arr[3][
               #      3]) / 4
               #  id0_std = np.std(id_ents_all_arr[0:4], 0)
               #  acc0 = (accuracy_all_arr[0] + accuracy_all_arr[1] + accuracy_all_arr[2] + accuracy_all_arr[3]) / 4
               #  acc0_std = np.std(accuracy_all_arr[0:4], 0)
               #  id1 = (id_ents_all_arr[0 + 4][3] + id_ents_all_arr[1 + 4][3] + id_ents_all_arr[2 + 4][3] +
               #         id_ents_all_arr[3 + 4][3]) / 4
               #  id1_std = np.std(id_ents_all_arr[0 + 4:4 + 4], 0)
               #  acc1 = (accuracy_all_arr[0 + 4] + accuracy_all_arr[1 + 4] + accuracy_all_arr[2 + 4] + accuracy_all_arr[
               #      3 + 4]) / 4
               #  acc1_std = np.std(accuracy_all_arr[0 + 4:4 + 4], 0)
               #  id2 = (id_ents_all_arr[0 + 8][3] + id_ents_all_arr[1 + 8][3] + id_ents_all_arr[2 + 8][3] +
               #         id_ents_all_arr[3 + 8][3]) / 4
               #  id2_std = np.std(id_ents_all_arr[0 + 8:4 + 8], 0)
               #  acc2 = (accuracy_all_arr[0 + 8] + accuracy_all_arr[1 + 8] + accuracy_all_arr[2 + 8] + accuracy_all_arr[
               #      3 + 8]) / 4
               #  acc2_std = np.std(accuracy_all_arr[0 + 8:4 + 8], 0)
               #  id3 = (id_ents_all_arr[0 + 12][3] + id_ents_all_arr[1 + 12][3] + id_ents_all_arr[2 + 12][3] +
               #        id_ents_all_arr[3 + 12][3]) / 4
               # id3_std = np.std(id_ents_all_arr[0 + 12:4 + 12], 0)
               # acc3 = (accuracy_all_arr[0 + 12] + accuracy_all_arr[1 + 12] + accuracy_all_arr[2 + 12] + accuracy_all_arr[
               #     3 + 12]) / 4
               # acc3_std = np.std(accuracy_all_arr[0 + 12:4 + 12], 0)
               #
               #  plt.plot(id0, label='ID-Entropy (p=0.2)')
               #  plt.fill_between(range(14), id0 - (id0_std / 2), id0 + (id0_std / 2), alpha=0.2)
               #  plt.plot(acc0, label='Test Accuracy (p=0.2)')
               #  plt.fill_between(range(14), acc0 - (acc0_std / 2), acc0 + (acc0_std / 2), alpha=0.2)
               #  plt.plot(id1, label='ID-Entropy (p=0.2)')
               #  plt.fill_between(range(14), id1 - (id1_std / 2), id1(id1_std / 2), alpha=0.2)
               #  plt.plot(acc1, label='Test Accuracy (p=0.2)')
               #  plt.fill_between(range(14), acc1 - (acc1_std / 2), acc1 + (acc1_std / 2), alpha=0.2)
               #  plt.plot(id2, label='ID-Entropy (p=0.2)')
               #  plt.fill_between(range(14), id2 - (id2_std / 2), id2 + (id2_std / 2), alpha=0.2)
               #  plt.plot(acc2, label='Test Accuracy (p=0.2)')
               #  plt.fill_between(range(14), acc2 - (acc2_std / 2), acc2 + (acc2_std / 2), alpha=0.2)
               #  plt.legend()
               #  plt.show()

                print('attws')
                print('attws')

                # torch.save(net, './stl10_rotoscale' + str(idx) + '.pt')
                # torch.save(net, './stl10_scale_steergroup_cnn_small' + str(idx) + '.pt')

                # net = torch.load('./rotoscale_mnifstscale_5rots_' + str(idx) + '.pt')

                accuracy = test_network(net, testloader, test_labels)
                # print("Test:", accuracy)
                accuracy = test_network(net, trainloader_small, train_labels)
                # print("Train:", accuracy)
                # accuracy_train = test_network(net,trainloader_small,train_labels)

                # f.write("Train:" + str(accuracy_train) + '\t' + "Test:" + str(accuracy) + '\n')
                # accuracy_all[idx, j] = accuracy


        print("Mean Accuracies of Networks:", np.mean(accuracy_all, 1))
        # f.write("Mean Accuracies of Networks:\t" + str(np.mean(accuracy_all, 1)) + '\n')
        # print("Standard Deviations of Networks:", np.std(accuracy_all, 0))
    # f.close()
