import numpy as np
from matplotlib import pyplot as plt
import pytorch_lightning as pl
import torch 
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from glob import glob
from scipy.integrate import odeint
import h5py
from pathlib import Path
from sklearn.decomposition import PCA
import sys
from conjnet import Encoder, Decoder, Evolver, ConjNet
from KS import KS
import seaborn as sns
color_palette = sns.color_palette("colorblind")

import train_conjnet_KS as experiment

ks = KS(L=22.0, K=15, dt=0.01)

dim = ks.d
mu_max = 0.04788123884905866 # Lyapunov exp.
dt = 0.01

def read_parameters(parameters_path):
    """
    read the parameters file into a dict
    """
    parameters_path=Path(parameters_path)
    pars = {} 
    pars['root_dir'] = str(parameters_path.parent)
    with open(parameters_path, 'r') as file:
        for line in file:
            split_line = line.split()
            try:
                pars[split_line[0]] = float(split_line[-1])
            except: 
                pars[split_line[0]] = split_line[-1]
    pars['datadir'] = pars['expparent'] + '/data'
    pars['ipo'] = int(pars['ipo'])
    pars['width'] = int(pars['width'])
    pars['nhidden'] = int(pars['nhidden'])
    return pars


def visualize_data(data, ipo, pos, ni=0, nf=1000, ax=None, c_zoom=0.8):
    """
    visualize the training dataset. 
    data should either be a FlowDataset object or a hdf5 file
    """
    
    if not(ax):
        fig = plt.figure(figsize=(6,6))
        ax = plt.subplot(projection='3d')
    
    po = list(pos.keys())[ipo-1]
    try: 
        pca = pos[po]['pca']
    except:
        pca = PCA(n_components=3)
        pca.fit(experiment.pos[po]['sol_'])
        pos[po]['sol_pca'] = pca.transform(experiment.pos[po]['sol_'])
        pos[po]['pca'] = pca

    ax.plot(
        pos[po]['sol_pca'][:, 0], 
        pos[po]['sol_pca'][:, 1], 
        pos[po]['sol_pca'][:, 2], 
        lw=4.0, 
        color=color_palette[ipo-1], 
        zorder=100, 
        alpha=1
    )
    
    try:
        xi_train, tt = data.trajectories[ni]
        dataset=True
    except: 
        dataset=False
        
    if dataset:
        for seed in range(ni,nf):
            xi_, _ = data.trajectories[seed]
            xi_pca = pca.transform(xi_)
            ax.plot(xi_pca[:, 0], xi_pca[:, 1], xi_pca[:, 2], 
                    color='black', alpha=0.05, lw=1.5)
    else: 
        f = h5py.File(data, 'r')
        for seed in range(ni,nf):
            xi_ = f[str(seed)]['xi_t_'][()]
            xi_pca = pca.transform(xi_)
            ax.plot(xi_pca[:, 0], xi_pca[:, 1], xi_pca[:, 2], 
                    color='black', alpha=0.05, lw=1.5)
        f.close()        


    ax.view_init(15, 145)
    ax.set_xlim(
        ax.get_xlim()[0] * c_zoom, 
        ax.get_xlim()[1] * c_zoom
        )
    ax.set_ylim(
        ax.get_ylim()[0] * c_zoom, 
        ax.get_ylim()[1] * c_zoom
        )
    ax.set_zlim(
        ax.get_zlim()[0] * c_zoom, 
        ax.get_zlim()[1] * c_zoom
        )

    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.set_xlabel("$PC_1$")
    ax.set_ylabel("$PC_2$")
    ax.set_zlabel("$PC_3$")    
    

def get_dataset(parameters, pos=experiment.pos):
    data_path=parameters['data_path']
    ipo=parameters['ipo']
    ctrans=parameters['ctrans']
    normpert=parameters['normpert']
    
    po = list(pos.keys())[ipo-1]
    
    mu=experiment.pos[po]['mu'] 
    om=experiment.pos[po]['om'] 
    mu_s = experiment.pos[po]['mu_s'] 
    T_p=experiment.pos[po]['T_p']
    
    dataset = experiment.FlowDataset(0, 2000, data_path=data_path, ipo = ipo, 
                                    transient = ctrans / mu, horizon = T_p,
                                    normpert=normpert)
    return dataset


def get_conjnet(parameters, pos=experiment.pos, ckpt=None):
    ipo = parameters['ipo']
    po = list(pos.keys())[ipo-1]
    width = parameters['width']
    nhidden = parameters['nhidden']
    root_dir = parameters['root_dir']
    try:
        kom = parameters['kom']
    except:
        print("no kom parameter found, setting kom=0")
        kom = 0
    
    mu=experiment.pos[po]['mu'] 
    om=experiment.pos[po]['om'] 
    mu_s = experiment.pos[po]['mu_s'] 
    T_p=experiment.pos[po]['T_p']
    
    om += kom * 2 * np.pi / T_p
    
    if ckpt==None:
#         ckpt = sorted(Path(log_dir).glob('**/*.ckpt'))[-1]
        ckpt = experiment.get_last_checkpoint(root_dir)
    
    parameters['checkpoint'] = ckpt
    
    tt_p = np.arange(0, T_p, experiment.dt)
    sol_po_ = experiment.pos[po]['sol_']
    po_torch = torch.zeros((tt_p.shape[0], experiment.d+1))
    po_torch[:, 0] = torch.tensor(tt_p)
    po_torch[:, 1:] = torch.tensor(sol_po_)
    
    encoder = Encoder(dim=dim, width=width, nhidden=nhidden)
    decoder = Decoder(dim=dim, width=width, nhidden=nhidden)
    evolver = Evolver(
        mu=mu, 
        om=om, 
        mu_s=mu_s, 
        T=T_p
    )
    conjnet = ConjNet(encoder,decoder,evolver,po = po_torch)

    checkpoint = torch.load(
        ckpt, 
        map_location=torch.device('cpu')
        )
    conjnet.load_state_dict(checkpoint['state_dict'])
    return conjnet


def visualize_predictions(
    parameters, seeds, dataset, conjnet, pos=experiment.pos, linestyle='-'
):
    ipo = parameters['ipo']
    po = list(pos.keys())[ipo-1]
    try: 
        pca = pos[po]['pca']
    except:
        pca = PCA(n_components=3)
        pca.fit(experiment.pos[po]['sol_'])
        pos[po]['sol_pca'] = pca.transform(experiment.pos[po]['sol_'])
        pos[po]['pca'] = pca
    
    
    nseeds = len(seeds)
    fig, axes = plt.subplots(nseeds, 2, figsize=(16,nseeds * 2))

    for iseed, seed in enumerate(seeds):
        try:
            xi_test, tt = dataset.trajectories[seed]
        except:
            xi_test, tt = dataset.generate_data(seed)

        tt = np.arange(0, dataset.horizon, 0.01)

        indata = torch.tensor(np.insert(np.ones((len(tt), dim)) * xi_test[0], 0, tt, axis=1))
        prediction = conjnet(indata.double())
        xi_pred = prediction.detach().cpu().numpy()
        xi_pred_pca = pca.transform(xi_pred)
        xi_test_pca = pca.transform(xi_test)

        for ax in axes[iseed, :]:

            ax.plot(pos[po]['sol_pca'][:, 0], pos[po]['sol_pca'][:,1], '--')

            ax.plot(xi_test_pca[:, 0], xi_test_pca[:, 1], label='test', linestyle=linestyle)
            ax.plot(xi_test_pca[0, 0], xi_test_pca[0, 1], 'k.', ms=10)
            ax.plot(xi_pred_pca[:, 0], xi_pred_pca[:, 1], label='pred', linestyle=linestyle)
            ax.plot(xi_pred_pca[0, 0], xi_pred_pca[0, 1], 'k.', ms=10)

        ic = xi_test_pca[0, :]
        ax.set_xlim(ic[0] - 0.1, ic[0] + 0.1)
        ax.set_ylim(ic[1] - 0.1, ic[1] + 0.1)
        ax.legend()
        
        
