from __future__ import print_function, division
import warnings
import random
import argparse
import numpy as np
from sklearn.cluster import KMeans

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from torchvision.utils import save_image

from utils import MnistDataset,FashionMnistDataset, USPSDataset, cluster_acc
from torch.autograd import Variable
from sklearn.decomposition import PCA
from scipy.stats import lognorm
import math
from torch.distributions import normal
torch.manual_seed(100)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import matplotlib
matplotlib.use('Agg')
from sklearn.manifold import TSNE
import time
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
class MVAE(nn.Module):
    def __init__(self,n_z):
        super(MVAE, self).__init__()
        self.conv_stack_encode = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ELU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ELU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5)
            )
        self.mlp_encode = nn.Sequential(
            nn.ELU(), 
            nn.Linear(256, n_z)
        )
        self.mlp_decode = nn.Sequential(
            nn.Linear(n_z, 256),
            nn.ELU(), 
        )
        self.conv_stack_decode = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=5, padding=4),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=2),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=2),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=2),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.conv_stack_encode(x)
        z = self.mlp_encode(x.view(x.size(0), -1))

        x_hat = self.mlp_decode(z)
        x_hat = self.conv_stack_decode(x_hat.view(x_hat.size(0),256,1,1))
        return x_hat, z
def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd
	
class MMVDEC(nn.Module):

    def __init__(self,
                 n_z,
                 n_clusters,
                 alpha=1,
                 pretrain_path='data/mvae_mnist_single.pkl'):        
        super(MMVDEC, self).__init__()
        self.alpha = 1.0
        self.pretrain_path = pretrain_path

        self.mvae = MVAE(n_z)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self, path=''):
        if path == '':
            pretrain_mvae(self.mvae)
        # load pretrain weights
        self.mvae.load_state_dict(torch.load(self.pretrain_path))
        print('load pretrained mvae from', path)

    def forward(self, x):

        x_bar, z = self.mvae(x)
        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x_bar, q


def target_distribution(q):
    weight = q**2 / q.sum(0) 

    return (weight.t() / weight.sum(1)).t()


def pretrain_mvae(model): #FOR MINIBATCH TRAINING
    '''
    pretrain autoencoder
    '''
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    data = dataset.x_tr[0:50]
    data = torch.Tensor(data).to(device)
    n_images = 16
    orig_images = data[:n_images]
    optimizer = Adam(model.parameters(), lr=args.lr)
    model.cuda()
    model.to(device)
    model.train()
    distribution = normal.Normal(0.0,math.sqrt(0.5))
    
    i = -1
    for epoch in range(200): #200
        
        train_loss = 0
        for batch_idx, (x, y,_) in enumerate(train_loader):
            i+=1
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            x_bar, z12 = model(x)

            true_samples = Variable(
                distribution.sample((args.batch_size, args.n_z)),
                requires_grad=False
                ).to(device)
                      
            BCE = torch.nn.functional.binary_cross_entropy(x_bar, x.view(-1, 1,28,28), reduction='mean')
            mmd = compute_mmd(true_samples,z12)

            loss = BCE+mmd
            loss.backward()
            train_loss += loss.item() 
            optimizer.step()
            torch.cuda.empty_cache()
            if i%100 ==0:
                print("BCE is {:.5f}, mmd loss is {:.5f} ".format(
                    BCE, mmd))
                x_bar,_ = model(data)
                decoded_images = x_bar.view(-1, 1, 28, 28)[:n_images]

                comparison = torch.cat([orig_images, decoded_images])    
                save_image(comparison.cpu(), 'data/results_mmvdec_single/' +'valid_mvae_mmvdec'+ str(epoch) +  '.png', nrow=n_images)  
        print("epoch {} loss={:.4f}".format(epoch,
                                            train_loss / (batch_idx + 1)))

      
    torch.save(model.state_dict(), args.pretrain_path)
    print("model saved to {}.".format(args.pretrain_path))
    torch.cuda.empty_cache() 
    
def partial_mvae(model,data,partial_size):          #FOR MINIBATCH REFERENCE
    model.eval()    #Caution of Batch Normalization
    hidden = torch.tensor([])
    data_batch = data.shape[0]
    
    m = int(data_batch/partial_size)
    n = data_batch%partial_size    
    for i in range(m):
        partial_data = data[i*partial_size:(i+1)*partial_size]
        x_bar_batch, hidden_batch = model.mvae(partial_data) #mvae from mmvdec model
        hidden = torch.cat((hidden, hidden_batch.data.cpu()))
    if n>0:    
        partial_data = data[m*partial_size:]
        x_bar_batch, hidden_batch = model.mvae(partial_data)
        hidden = torch.cat((hidden, hidden_batch.data.cpu()))        
    return hidden

def partial_model(model,data,partial_size):  #FOR MINIBATCH REFERENCE
    model.eval()    #Caution of Batch Normalization
    x_bar = torch.tensor([])
    tmp_q = torch.tensor([])
    m = int(data.size(0)/partial_size)
    n = data.size(0)%partial_size    
    for i in range(m):
        partial_data = data[i*partial_size:(i+1)*partial_size]
        partial_data = partial_data.to(device)
        x_bar_batch, tmp_q_batch = model(partial_data)
        x_bar = torch.cat((x_bar, x_bar_batch.data.cpu()))
        tmp_q = torch.cat((tmp_q, tmp_q_batch.data.cpu())) 
    if n>0:
        partial_data = data[m*partial_size:]
        partial_data = partial_data.to(device)
        x_bar_batch, tmp_q_batch = model(partial_data)
        x_bar = torch.cat((x_bar, x_bar_batch.data.cpu()))
        tmp_q = torch.cat((tmp_q, tmp_q_batch.data.cpu())) 
        torch.cuda.empty_cache()     
    return x_bar, tmp_q


def train_mmvdec():

    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning) 

    model = MMVDEC(
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        alpha=1.0,
        pretrain_path=args.pretrain_path).to(device)
    model.pretrain()
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # cluster parameter initiate
    data = dataset.x_tr
    y = dataset.y_tr
    data = torch.Tensor(data).to(device)

    partial_size = 5000
    
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)

    kmean_batch = 5000                   
    hidden12 = partial_mvae(model,data,kmean_batch)       
    y12_pred = kmeans.fit_predict(hidden12.data.cpu().numpy()) 
    nmi_k12 = nmi_score(y12_pred, y)    
    acc12 = cluster_acc(y, y12_pred)
    ari12 = ari_score(y, y12_pred)
    print('KMeans : Acc {:.4f}'.format(acc12),"nmi score={:.4f}".format(nmi_k12), 'ari {:.4f}'.format(ari12))


    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device) 

    y_pred_last = y12_pred

    hidden12 = None    
    x_bar = None

    ep = 0
    predict_acc= []
    predict_nmi= []
    model.train()
    for epoch in range(100): 

        if epoch % args.update_interval == 0:
            
            _, tmp_q = partial_model(model,data,partial_size)

            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
            # evaluate clustering performance          
            y_pred = tmp_q.cpu().numpy().argmax(1)
            tmp_q = tmp_q.to(device)
   
            p = p.to(device)#            
              
            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]            
            y_pred_last = y_pred
            acc = cluster_acc(y, y_pred)
            nmi = nmi_score(y, y_pred)
            ari = ari_score(y, y_pred)
            #Plot curves
            predict_acc.append(acc)
            predict_nmi.append(nmi)
            ep = ep + args.update_interval
            print('Iter {}'.format(epoch), ':Acc {:.4f}'.format(acc),
                  ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))

            if epoch > 0 and delta_label < args.tol:
                print('delta_label {:.4f}'.format(delta_label), '< tol',
                      args.tol)
                print('Reached tolerance threshold. Stopping training.')
                break
                
                
        for batch_idx, (x_batch, y_batch,idx) in enumerate(train_loader):

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            idx = idx.type(torch.LongTensor).to(device)
            x_bar, q = model(x_batch)

            reconstr_loss = F.mse_loss(x_bar, x_batch)
            kl_loss = F.kl_div(q.log(), p[idx])

            loss =  kl_loss + reconstr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        torch.cuda.empty_cache() 
        
        
    t = str(random.randint(1,1000))
    plt.plot(range(0,ep,args.update_interval),predict_acc)
    plt.plot(range(0,ep,args.update_interval),predict_nmi)
    plt.savefig('./check_mmvdec_single/mnist_Predict_ACC_'+ t +'.pdf')
    plt.close()  
    torch.save(model.state_dict(), 'data/mmvdec_mnist_single.pkl')

from sklearn.manifold import TSNE
import time
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
N = 70000 #subset_number

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
RS = 123

def fashion_scatter(x, colors,idx,message):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    #sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    # show the text for digit corresponding to the true label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0) #true labels 
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    plt.savefig('./check_mmvdec_single/scatter__mmvdec'+ str(idx) +'_'+message+ '.pdf', bbox_inches='tight')
    plt.close()
    return f, ax, sc, txts


def test_mmvdec():
    

    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning) 
    
    
    model = MMVDEC(
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        alpha=1.0,
        pretrain_path=args.pretrain_path).to(device)

    model.load_state_dict(torch.load('data/mmvdec_mnist_single.pkl'))
    # cluster parameter initiate
    data = dataset.x_tr
    y = dataset.y_tr
    data = torch.Tensor(data).to(device)
    model.eval()
    hidden = partial_mvae(model,data,20)
    z_show = hidden[0:N]
    y_show = y[0:N]  
    fashion_tsne_hidden_without_pca = TSNE(random_state=RS).fit_transform(z_show.detach().cpu())
    t = str(random.randint(1,1000))
    fashion_scatter(fashion_tsne_hidden_without_pca, y_show,0,'post_trained_'+t) 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--pretrain_path', type=str, default='data/mnist')
    parser.add_argument(
        '--gamma',
        default=0.1, #0.1
        type=float,
        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.dataset == 'mnist':
        args.pretrain_path = 'data/mvae_mnist_single.pkl'
        args.n_clusters = 10
        args.n_input = 28*28
        dataset = MnistDataset()
    if args.dataset == 'famnist':
        args.pretrain_path = 'data/mvae_famnist_single.pkl'
        args.n_clusters = 10
        args.n_input = 28*28
        dataset = FashionMnistDataset()        
    if args.dataset == 'usps':
        args.pretrain_path = 'data/mvae_usps_single.pkl'
        args.n_clusters = 10
        args.n_input = 16*16
        dataset = USPSDataset()
       
    print(args)
    train_mmvdec()
    test_mmvdec()
