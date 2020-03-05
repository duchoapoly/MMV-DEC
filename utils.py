
from __future__ import division, print_function
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms
from torchvision import datasets
def load_mnist(path='./data/mnist.npz'):
    f = np.load(path)

    x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f[
        'x_test'], f['y_test']
    f.close()
    
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    x_train = x
    y_train = y
    x_tr = x_train
    y_tr = y_train
    x_tr = np.expand_dims(x_tr,axis = 1) #to be processed by convolution layer
    x_tr = x_tr.astype(np.float32)
    y_tr = y_tr.astype(np.int64)
    x_tr = np.divide(x_tr, 255.)
    print('MNIST training samples', x_tr.shape)
    x_te = x_test
    y_te = y_test
    x_te = np.expand_dims(x_te,axis = 1) #to be processed by convolution layer
    x_te = x_te.astype(np.float32)
    y_te = y_te.astype(np.int64)
    x_te = np.divide(x_te, 255.)
    print('MNIST testing samples', x_te.shape)
    return x_tr, y_tr 

class MnistDataset(Dataset):

    def __init__(self):
        self.x_tr, self.y_tr= load_mnist()
    def __len__(self):
        return self.x_tr.shape[0]

    def __getitem__(self, idx):
  
        return torch.from_numpy(np.array(self.x_tr[idx])), torch.from_numpy(
            np.array(self.y_tr[idx])), torch.from_numpy(np.array(idx))

class FashionMnistDataset(Dataset):
    def __init__(self):
        fashion_dataset_tr = datasets.FashionMNIST('/home/hoatran/IDEC-MVAE/', train=True, 
                                    transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    ]), 
                                    target_transform=None, 
                                    download=False)
        x_tr = fashion_dataset_tr.train_data
        x_tr = np.expand_dims(x_tr,axis = 1) #to be processed by convolution layer
        x_tr = np.divide(x_tr, 255.)
        y_tr = np.array(fashion_dataset_tr.train_labels)

        self.x_tr = x_tr.astype(np.float32)
        self.y_tr = y_tr.astype(np.int64)

        print('x_tr shape',self.x_tr.shape)
        print('y_tr shape',self.y_tr.shape)

    def __len__(self):
        return self.x_tr.shape[0]

    def __getitem__(self, idx):
  
        return torch.from_numpy(np.array(self.x_tr[idx])), torch.from_numpy(
            np.array(self.y_tr[idx])), torch.from_numpy(np.array(idx))
    
def load_usps(data_path='./data'):

    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    print('data list len',len(data))
    
    data = [line.split() for line in data]   
    data = np.array(data)

    data_train = data[:, 1:]
    labels_train = data[:, 0]    
    data_train = data_train.astype(np.float32)
    data_train = np.expand_dims(data_train,axis = 1) #to be processed by convolution layer
    data_train = np.reshape(data_train,[-1,1,16,16])
    labels_train = labels_train.astype(np.int64)
    
    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [line.split() for line in data]
    data = np.array(data)
    
    data_test = data[:, 1:]
    labels_test = data[:, 0]
    data_test = data_test.astype(np.float32)
    data_test = np.expand_dims(data_test,axis = 1) #to be processed by convolution layer 
    data_test = np.reshape(data_test,[-1,1,16,16])
    labels_test = labels_test.astype(np.int64)
    x = np.concatenate((data_train, data_test)).astype('float32') #float64
    #x = np.divide(x, 255.)
    y = np.concatenate((labels_train, labels_test))
    print( 'USPS samples', x.shape)
    return x, y


class USPSDataset(Dataset):

    def __init__(self):
        x_tr, y_tr= load_usps()
        x_tr = torch.from_numpy(x_tr)

        x_tr = torch.nn.functional.interpolate(x_tr,[28,28],mode='bilinear')
        self.x_tr = np.divide(x_tr, 2.)
        self.y_tr = y_tr
       
    def __len__(self):
        return self.x_tr.shape[0]

    def __getitem__(self, idx):
 
        return torch.from_numpy(np.array(self.x_tr[idx])), torch.from_numpy(
            np.array(self.y_tr[idx])), torch.from_numpy(np.array(idx))

def cluster_acc(y_true, y_pred):

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
