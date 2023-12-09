#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import sys
import torch
from codebase import utils as ut
import argparse
from pprint import pprint
cuda = torch.cuda.is_available()
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
import numpy as np
import math
import time
from torch.utils import data
from utils import get_batch_unin_dataset_withlabel, _h_A
import matplotlib.pyplot as plt
import random
import torch.utils.data as Data
from PIL import Image
import os
import numpy as np
from torchvision import transforms
from codebase import utils as ut
from codebase.models.mask_vae_pendulum import CausalVAE
import argparse
from pprint import pprint
cuda = torch.cuda.is_available()
from torchvision.utils import save_image

os.environ["CUDA_VISIBLE_DEVICE"] = "7"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--epoch_max',   type=int, default=101,    help="Number of training epochs")
parser.add_argument('--iter_save',   type=int, default=5, help="Save model every n epochs")
parser.add_argument('--run',         type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',       type=int, default=1,     help="Flag for training")
parser.add_argument('--color',       type=int, default=False,     help="Flag for color")
parser.add_argument('--toy',       type=str, default="pendulum_mask",     help="Flag for toy")
args = parser.parse_args()
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
def _sigmoid(x):
    I = torch.eye(x.size()[0]).to(device)
    x = torch.inverse(I + torch.exp(-x))
    return x
    
class DeterministicWarmup(object):
	"""
	Linear deterministic warm-up as described in
	[S?nderby 2016].
	"""
	def __init__(self, n=100, t_max=1):
		self.t = 0
		self.t_max = t_max
		self.inc = 1/n

	def __iter__(self):
		return self

	def __next__(self):
		t = self.t + self.inc

		self.t = self.t_max if t > self.t_max else t
		return self.t
layout = [
	('model={:s}',  'causalvae'),
	('run={:04d}', args.run),
	('color=True', args.color),
	('toy={:s}', str(args.toy))
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)
lvae = CausalVAE(name=model_name, z_dim=16).to(device)
if not os.path.exists('./figs_vae/'): 
	os.makedirs('./figs_vae/')

dataset_dir = './causal_data/pendulum'
train_dataset = get_batch_unin_dataset_withlabel(dataset_dir, 64)
test_dataset = get_batch_unin_dataset_withlabel(dataset_dir, 1)
optimizer = torch.optim.Adam(lvae.parameters(), lr=1e-3, betas=(0.9, 0.999))
beta = DeterministicWarmup(n=100, t_max=1) # Linear warm-up from 0 to 1 over 50 epoch



def save_model_by_name(model, global_step):
	save_dir = os.path.join('checkpoints', model.name)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
	state = model.state_dict()
	torch.save(state, file_path)
	print('Saved to {}'.format(file_path))
 
initial_A = lvae.dag.A.clone().detach().cpu().numpy()
#mean = np.mean(initial_A, axis=0, keepdims=True)
#std = np.mean(initial_A, axis=0, keepdims=True)
mean = np.mean(initial_A)
std = np.std(initial_A)
#initial_A = (initial_A - mean) / std
plt.imshow(initial_A, cmap="Blues", interpolation="nearest")
plt.colorbar()
plt.savefig("figs_vae/causal_matrix_initial.png")
plt.close()

"""pretrain_optimizer = optim.Adam(lvae.dag.parameters(), lr=1e-3)
prev_h_a = 0
lam = 0.3
c = 1
c_s = 1
l_u_s = []
for epoch in range(10):
    running_l_u = 0
    print("Epoch: {}".format(epoch))
    lvae.train()
    for u, l in train_dataset:
        l_mean = torch.mean(l, dim=0, keepdim=True)
        l_std = torch.std(l, dim=0, keepdim=True)
        l = (l - mean) / l_std
        pretrain_optimizer.zero_grad()
        l = l.to(device)
        dag_param = lvae.dag.A
        #L, kl, rec, reconstructed_image, _, l_u = lvae.negative_elbo_bound(u,l,sample = False)
        l = l.view(-1, l.size()[1], 1)
        l_u = nn.MSELoss()(torch.matmul(dag_param.t(), l) , l)
        h_a = _h_A(dag_param, dag_param.size()[0])
        L = l_u + lam * h_a + c / 2 * h_a * h_a
        L.backward()
        pretrain_optimizer.step()
        lam = lam + c_s * h_a.item()
        running_l_u += l_u.item()
    l_u_s.append(running_l_u)
    plt.plot(l_u_s)
    plt.savefig("l_u.png")
    plt.close()
    #print("prev_h_a = {}".format(prev_h_a))
    #print("h_a = {}".format(h_a))
    if abs(h_a) > 0.25 * abs(prev_h_a):
        c_s = 10 * c_s
    #print("c_s = {}".format(c_s))
    print("h_a = {}".format(h_a))
    prev_h_a = h_a
        
    
    visualization_diagram = dag_param.clone().detach().cpu().numpy()
    print(visualization_diagram)
    #mean = np.mean(visualization_diagram, axis=0, keepdims=True)
    #std = np.mean(visualization_diagram, axis=0, keepdims=True)
    mean = np.mean(visualization_diagram)
    std = np.mean(visualization_diagram)
    #visualization_diagram = (visualization_diagram - mean) / std
    plt.imshow(visualization_diagram, cmap="Blues", interpolation="nearest")
    plt.colorbar()
    plt.savefig("figs_vae/causal_matrix_{}.png".format(epoch))
    plt.close()"""
 
for epoch in range(args.epoch_max):
	lvae.train()
	total_loss = 0
	total_rec = 0
	total_kl = 0
	for u, l in train_dataset:
		optimizer.zero_grad()
		#u = torch.bernoulli(u.to(device).reshape(u.size(0), -1))
		#l_mean = torch.mean(l, dim=0, keepdim=True)
		#l_std = torch.std(l, dim=0, keepdim=True)
		#l = (l - mean) / std
		u = u.to(device)
		L, kl, rec, reconstructed_image, _, l_u = lvae.negative_elbo_bound(u,l,sample = False)

		dag_param = lvae.dag.A

		#dag_reg = dag_regularization(dag_param)
		h_a = _h_A(dag_param, dag_param.size()[0])
		L = L + 3*h_a + 0.5*h_a*h_a #- torch.norm(dag_param) 


		L.backward()
		optimizer.step()
		#optimizer.zero_grad()

		total_loss += L.item()
		total_kl += kl.item() 
		total_rec += rec.item() 

		m = len(train_dataset)
		save_image(u[0], 'figs_vae/reconstructed_image_true_{}.png'.format(epoch), normalize = True) 
		save_image(reconstructed_image[0], 'figs_vae/reconstructed_image_{}.png'.format(epoch), normalize = True) 

	if epoch % 1 == 0:
		print(str(epoch)+' loss:'+str(total_loss/m)+' kl:'+str(total_kl/m)+' rec:'+str(total_rec/m)+'m:' + str(m))

	if epoch % args.iter_save == 0:
		ut.save_model_by_name(lvae, epoch)

	#visualization_diagram = dag_param.clone().detach().cpu().numpy()
	#print(visualization_diagram)
	#mean = np.mean(visualization_diagram, axis=0, keepdims=True)
	#std = np.mean(visualization_diagram, axis=0, keepdims=True)
	#visualization_diagram = (visualization_diagram - mean) / std
	#plt.imshow(visualization_diagram, cmap="Blues", interpolation="nearest")
	#plt.colorbar()
	#plt.savefig("figs_vae/causal_matrix_{}.png".format(epoch))
	#plt.close()
 