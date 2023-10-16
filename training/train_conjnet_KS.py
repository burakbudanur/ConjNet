from argparse import ArgumentParser
import numpy as np
from matplotlib import pyplot as plt
import pytorch_lightning as pl
import torch 
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
import torch.nn.functional as F
from scipy.integrate import odeint
import h5py
import sys
from pathlib import Path
from multiprocessing import Pool
from numpy.linalg import eig, norm
sys.path.append('../')
from conjnet import Encoder, Decoder, Evolver, ConjNet
from KS import KS
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

torch.set_default_dtype(torch.float64)

ks = KS(L=22.0, K=15, dt=0.01)

d = ks.d
mu_max = 0.04788123884905866 # Lyapunov exp.
dt = 0.01

po_path = Path("../data/5POs_L22.000.hdf5")

fpos = h5py.File(po_path, 'r')
pos = {}

for ipo, po in enumerate(fpos.keys()):

    pos[po] = {}
    for key in fpos[po].keys():
        pos[po][key] = fpos[po][key][()]

    xi_p = pos[po]['xi_p']
    T_p = pos[po]['T_p']
    J_p = pos[po]['J_p']

    LA_p, _ = eig(J_p)

    i_sort = np.argsort(np.abs(LA_p))[::-1]
    LA_p = LA_p[i_sort]

    mu = np.log(np.abs(LA_p[0])) / T_p 

    if np.imag(LA_p[0]) == 0 and np.real(LA_p[0]) < 0:
        # non-orientable neighborhood
        om = np.abs(np.angle(LA_p[0])) / T_p

        i_mu_s = np.argwhere(np.abs(LA_p) - 1 < -1e-3).reshape(-1)[0]
        mu_s = np.log(np.real(np.abs(LA_p[i_mu_s]))) / T_p 
    elif not(np.angle(LA_p[0]) == 0):
        # spiraling neighborhood
        om = np.abs(np.angle(LA_p[0])) / T_p
        mu_s = 0
    else:
        # expanding/contracting neighborhood
        om = 0
        i_mu_s = np.argwhere(np.abs(LA_p) - 1 < -1e-3).reshape(-1)[0]
        mu_s = np.log(np.real(np.abs(LA_p[i_mu_s]))) / T_p 

    pos[po]['mu'] = mu # these will go into Evolver()
    pos[po]['om'] = om 
    pos[po]['mu_s'] = mu_s 

    tt = np.arange(0, T_p, dt)
    sol = odeint(ks.rhs, xi_p, tt)
    sol_ = ks.symmreduce(sol)
    pos[po]['sol'] = sol
    pos[po]['sol_'] = sol_ 

fpos.close()

class FlowDataset(Dataset):
    def __init__(self, ni, nf, ipo, data_path, transient, horizon, normpert):
        
        if not(ni < nf):
            raise ValueError("ni < nf must be satisfied.")

        self.seeds = np.arange(ni, nf)
        self.ipo = ipo
        self.transient = transient
        self.horizon = horizon
        self.normpert = normpert

        xi_t, tt = self.generate_data(seed=self.seeds[0])

        self.lentraj = xi_t.shape[0]
        self.numseeds = len(self.seeds)
        self.data_path = data_path

        try:
            f = h5py.File(data_path, 'r')
            self.trajectories = {}
            for key in f.keys():
                if int(key) in self.seeds:
                    self.trajectories[int(key)] = (
                        f[key]['xi_t_'][()], f[key]['tt'][()]
                        )
            f.close()
        except:
            self.trajectories = {}
            f = h5py.File(self.data_path, 'w')
            f.close()

    def __len__(self):
        return self.numseeds * self.lentraj

    def __getitem__(self, idx):

        seed = idx // self.lentraj + self.seeds[0]
        ntraj = idx % self.lentraj

        try:
            # read data off if the trajectory is there
            xi_t, tt = self.trajectories[seed]
        except:
            # compute the trajectory if it's not there
            self.trajectories[seed] = self.generate_data(seed=seed)
            # save trajectory for use in the next run
            f = h5py.File(self.data_path, 'r+')
            f.create_group(str(seed))
            f[str(seed)]['xi_t_'] = self.trajectories[seed][0]
            f[str(seed)]['tt'] = self.trajectories[seed][1]
            f.close()
            xi_t, tt = self.trajectories[seed]

        data_in = np.concatenate(([tt[ntraj]], xi_t[0, :]))
        data_out = xi_t[ntraj, :]

        data_in = torch.tensor(data_in).double()
        data_out = torch.tensor(data_out).double()

        return data_in, data_out

    def generate_data(self, ipo=None, transient=None, horizon=None, seed=None, 
                      normpert=None, pos=pos):
        if not(ipo):
            ipo = self.ipo
        if not(transient):
            transient=self.transient
        if not(horizon):
            horizon = self.horizon
        if not(normpert):
            normpert = self.normpert
        
        rng = np.random.default_rng(seed=seed)

        po = list(pos.keys())[ipo-1]
        sol_po = pos[po]['sol']

        n_sol = rng.integers(0, sol_po.shape[0])
        xi_p = sol_po[n_sol, :]

        de_xi  = rng.standard_normal(d) * np.abs(xi_p)
        de_xi *= normpert / norm(de_xi)

        de_xi *= xi_p

        xi_0 = xi_p + de_xi
        xi_0 = ks.flow(xi_0, transient)
        tt = np.arange(0, horizon, dt)

        xi_t = odeint(ks.rhs, xi_0, tt)
        xi_t_ = ks.symmreduce(xi_t)

        return xi_t_, tt

    
class NullDataset(Dataset):
    def __init__(self, num_data, dim):
        self.num_data = num_data        
        self.dim = dim        

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):

        data_in = torch.zeros((self.dim+1))
        data_out = torch.zeros((self.dim))

        return data_in, data_out
    

def get_last_checkpoint(root_dir):
    """
    Given a root_dir, we return a checkpoint path in 
    
        root_dir/lightning_logs/version_k/*epoch=m-*.ckpt
    
    such that k is maximum and m is maximum given k. 
    """
    
    log_dir = Path(root_dir) / "lightning_logs"
    version_dirs = sorted(Path(log_dir).glob('version*'))

    versions = []
    last_version = 0 
    for i, path in enumerate(version_dirs):
        pathstr = str(path)
        for cursor in range(len(pathstr) - 6):
            if pathstr[cursor:cursor+7] == 'version':
                versionstr = pathstr[cursor:]
                if not(versionstr in versions):
                    versions.append(versionstr)

                    if int(versionstr[8:]) > last_version:
                        last_version = int(versionstr[8:])

    ckptdir = Path(log_dir) / f'version_{last_version}/checkpoints'
    ckptlist = sorted(ckptdir.glob("**/*.ckpt"))

    last_epoch = 0

    for i, path in enumerate(ckptlist):
        pathstr = str(path)
        for cursor in range(len(pathstr)):
            if pathstr[cursor:cursor+len("epoch")] == "epoch":
                for cursor_ in range(cursor, cursor+20):
                    if pathstr[cursor_] == "-":
                        break
                epochstr = pathstr[cursor:cursor_]
                break

        i_epoch = int(epochstr[len("epoch="):])
        if i_epoch > last_epoch:
            last_epoch = i

    last_checkpoint = ckptlist[last_epoch]
    
    return last_checkpoint

    
    
def train_conjnet(**kwargs): 

    expparent=Path(kwargs['expparent'])
    expparent.mkdir(parents=True, exist_ok=True)

    ipo = kwargs['ipo']
    ctrans = kwargs['ctrans']
    learnrate = kwargs['learnrate']
    checkpoint = kwargs['checkpoint']
    normpert = kwargs['normpert']
    nhidden=kwargs['nhidden']
    width=kwargs['width']
    maxepochs=kwargs['maxepochs']
    parallel_arg = kwargs['parallel_arg']
    poloss = kwargs['poloss']

    po = list(pos.keys())[ipo - 1]
    mu = pos[po]['mu']
    om = pos[po]['om']
    mu_s = pos[po]['mu_s']
    T_p = pos[po]['T_p']
    
    transient = ctrans / mu
    
    data_parent = expparent / "data"
    data_parent.mkdir(exist_ok=True)
    data_path = data_parent  / f"po_{ipo}_trajs_ctrans_{ctrans:.2f}_normpert_{normpert:.6f}.hdf5"
    kwargs['data_path'] = data_path
    
    training_data = FlowDataset(
        0, 1000, ipo=ipo, data_path=data_path, transient=transient, 
        horizon=T_p, normpert=normpert
    )
    testing_data = FlowDataset(
        1000,2000, ipo=ipo, data_path=data_path, transient=transient,
        horizon=T_p, normpert=normpert
    )

    # use 20% of training data for validation
    train_set_size = int(len(training_data) * 0.8)
    valid_set_size = len(training_data) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = random_split(
        training_data, [train_set_size, valid_set_size], generator=seed
        )

    train_dataloader = DataLoader(train_set, batch_size=100, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=100, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=100, shuffle=False)
    
    # subdir for logging named after the argument varied and its value
    if type(kwargs[parallel_arg]) == int:
        root_dir = expparent / f"{parallel_arg}_{kwargs[parallel_arg]}/"
    else:
        root_dir = expparent / f"{parallel_arg}_{kwargs[parallel_arg]:.2e}/"
        # assuming we will not vary these beyond the second decimal point

    root_dir.mkdir(exist_ok=True)
    path_parameters = root_dir / "parameters.txt"

    f = open(path_parameters, "w")
    for key in kwargs:
        f.write(f"{key} = {kwargs[key]} \n")
    f.close()

    encoder = Encoder(dim=d, width=width, nhidden=nhidden)
    decoder = Decoder(dim=d, width=width, nhidden=nhidden)
    evolver = Evolver(mu=mu, om=om, mu_s = mu_s, T=T_p)

    if poloss:
        tt_p = np.arange(0, T_p, dt)
        
        sol_po_ = pos[po]['sol_']
        po_torch = torch.zeros((tt_p.shape[0], d+1))
        po_torch[:, 0] = torch.tensor(tt_p)
        po_torch[:, 1:] = torch.tensor(sol_po_)
        conjnet = ConjNet(
            encoder=encoder, 
            decoder=decoder, 
            evolver=evolver, 
            lr=learnrate,
            po = po_torch.cuda()
            )
    else:
        conjnet = ConjNet(
            encoder=encoder, 
            decoder=decoder, 
            evolver=evolver, 
            lr=learnrate
            )

    if not(checkpoint==''):
        print("Checkpoint given: ", checkpoint)

        checkpoint = torch.load(
            checkpoint, 
            map_location=torch.device('cuda')
            )
        conjnet.load_state_dict(checkpoint['state_dict'])

    else:
        try:
            
#             last_checkpoint = sorted(root_dir.glob('**/*.ckpt'))[-1]
            log_dir = root_dir
            last_checkpoint = get_last_checkpoint(log_dir)
    
            print("Checkpoint found: ", last_checkpoint)
            
            checkpoint = torch.load(
                last_checkpoint, 
                map_location=torch.device('cuda')
                )
            
            conjnet.load_state_dict(checkpoint['state_dict'])

        except:
            print("no checkpoint found")
            
            if poloss:
            
                null_training_data = NullDataset(1000, 30)
                null_loader = DataLoader(null_training_data)
                null_valid_loader = DataLoader(null_training_data)
                null_trainer = pl.Trainer(
                    accelerator='gpu', 
                    max_epochs=maxepochs, 
                    default_root_dir=root_dir,
                    precision=64,
                    callbacks=[
                        EarlyStopping(monitor="val_loss",  patience=3, mode="min", min_delta=1e-6), 
                        ModelCheckpoint(
                            every_n_epochs=1,
                            save_on_train_epoch_end=True,
                            save_top_k=-1 # save all models
                        )           
                    ],
                    check_val_every_n_epoch=10, 
                    enable_progress_bar=False
                )
                null_trainer.fit(conjnet, null_loader, null_valid_loader)
    
    trainer = pl.Trainer(
        accelerator='gpu', 
        max_epochs=maxepochs, 
        default_root_dir=root_dir, 
        precision=64,
        callbacks=[ 
            EarlyStopping(monitor="val_loss",  patience=3, mode="min", min_delta=1e-6), 
            ModelCheckpoint(
                every_n_epochs=1,
                save_on_train_epoch_end=True,
                save_top_k=-1 # save all models
            )
        ],
        check_val_every_n_epoch=10, 
        enable_progress_bar=True  
        )

    trainer.fit(conjnet, train_dataloader, valid_loader)
    # test the model
    trainer.test(model=conjnet, dataloaders=test_dataloader)

    return

 
if __name__ == "__main__":
        
    parser = ArgumentParser()
    parser.add_argument(
        '--expparent', action='store', type=str, default='./results', 
        help='Parent dir for results.'
        )
    parser.add_argument(
        '--nproc', action='store', type=int, default=5, 
        help='Number of processors.'
        )
    parser.add_argument(
        '--maxepochs', action='store', type=int, default=100, 
        help='Maximum number of epochs.'
        )
    parser.add_argument(
        '--checkpoint', action='store', type=str, default='',
        help='Checkpoint to start the training from.'
    )
    parser.add_argument(
        '--poloss', action='store', type=bool, default=True, 
        help='Include periodic orbit loss term.'
    )
    parser.add_argument(
        '--ipo', nargs='+', action='store', type=int, default=[1,2,3,4,5], 
        choices=range(1,6), help='Which pos (1,...,5) train the ConjNet for.'
        )
    parser.add_argument(
        '--ctrans', nargs='+', action='store', type=float, default=[2.0], 
        help='transient = ctrans / mu, where mu is the leading Floquet exp..'
        )
    parser.add_argument(
        '--learnrate', nargs='+', action='store', type=float, default=[1e-4], 
        help='Learning rates.'
    )
    parser.add_argument(
        '--normpert', nargs='+', action='store', type=float, default=[1e-3], 
        help='Perturbation amplitude.'
    )
    parser.add_argument(
        '--nhidden', nargs='+', action='store', type=int, default=[1], 
        help='Depth of the encoder/decoder -- not yet implemented.'
    )
    parser.add_argument(
        '--width', nargs='+', action='store', type=int, default=[128], 
        help='Width of the encoder/decoder -- not yet implemented.'
    )

    args = parser.parse_args()
    argsdict = vars(args)

    nproc = argsdict["nproc"]

    parallel = False
    pars = {}
    for key in argsdict:

        if type(argsdict[key]) == type([]):
            lenarg = len(argsdict[key])
            if lenarg > 1 and not(parallel):
                parallel=True 
                parallel_arg = key
                parallel_vals = argsdict[key]

                if not(lenarg == nproc):
                    raise ResourceWarning(
                        "nproc is not equal to the rank of parallelized argument."
                        )
                continue

            elif lenarg > 1 and parallel:
                raise ValueError(
                    "Only one parameter can be parallelized in an experiment."
                    )
            else:
                pars[key]=argsdict[key][0]
        else:
            pars[key]=argsdict[key]
    
    if not(parallel):
        parallel_arg = 'ipo'
        parallel_vals = argsdict['ipo']

    def train_wrapper(parallel_val):
        wrapper_pars = dict(pars)
        wrapper_pars[parallel_arg] = parallel_val
        wrapper_pars['parallel_arg'] = parallel_arg
        return train_conjnet(**wrapper_pars)

    with Pool(nproc) as p:
        p.map(train_wrapper, parallel_vals)
