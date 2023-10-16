import numpy as np
import pytorch_lightning as pl
import torch 
from torch import nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

class Encoder(nn.Module):
    def __init__(self, dim=3, width=128, nhidden=1):
        super().__init__()
        self.dim = dim
        self.encoder = nn.Sequential(
            nn.Linear(self.dim, width),
            nn.SiLU()
        )

        for i in range(nhidden):
            self.encoder.append(nn.Linear(width, width))
            self.encoder.append(nn.SiLU())

        self.encoder.append(nn.Linear(width, 3))
    
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, dim=3, width=128, nhidden=1):
        super().__init__()
        self.dim = dim
        self.decoder = nn.Sequential(
            nn.Linear(3, width),
            nn.SiLU()
        )

        for i in range(nhidden):
            self.decoder.append(nn.Linear(width, width))
            self.decoder.append(nn.SiLU())

        self.decoder.append(nn.Linear(width, self.dim))
    
    def forward(self, x):
        return self.decoder(x)


class Evolver(nn.Module):
    
    def __init__(
            self, mu=None, om=None, mu_s = 0, T=2*np.pi
            ):
        super().__init__()
        if not(mu is None):
            self.mu = mu
        else:
            self.mu = torch.nn.Parameter(torch.rand(())) # trainable mu

        self.flip = False
        if not(om is None):
            self.om = om
            try:
                self.flip = np.real(np.exp(1j * om * T) + 1) < 1e-7
            except: 
                self.flip = np.real(np.exp(1j * om.detach().numpy() * T) + 1) < 1e-7                
        else:
            self.om = torch.nn.Parameter(torch.rand(())) # trainable om

        self.mu_s = mu_s
        self.T = T

        self.spiral = not(self.om == 0) and not(self.flip)

    def forward(self, x):
        self.t = x[:, 0]
        self.xi_0 = x[:, 1:] 
        self.batch_size = len(self.t)

        self.th_0 = torch.atan2(self.xi_0[:, 1], self.xi_0[:, 0])
        
        # frame fixing
        self.xi_0_hat = torch.clone(self.xi_0)
        self.xi_0_hat[:, 0] = (
            torch.cos(-self.th_0) * self.xi_0[:, 0] 
            - torch.sin(-self.th_0) * self.xi_0[:, 1]
        )
        self.xi_0_hat[:, 1] = (
            torch.sin(-self.th_0) * self.xi_0[:, 0] 
            + torch.cos(-self.th_0) * self.xi_0[:, 1]
        )

        self.de_xi_0 = self.xi_0_hat
        self.de_xi_0[:, 0] -= 1.0

        if self.spiral:
            # Spiraling neighborhood
            self.de_xi_t = torch.clone(self.de_xi_0)
            self.de_xi_t[:, 0] = torch.exp(self.mu * self.t) * (
                torch.cos(self.om * self.t) * self.de_xi_0[:, 0]
                + torch.sin(self.om * self.t) * self.de_xi_0[:, 2]
            )
            self.de_xi_t[:, 2] = torch.exp(self.mu * self.t) * (
                - torch.sin(self.om * self.t) * self.de_xi_0[:, 0]
                + torch.cos(self.om * self.t) * self.de_xi_0[:, 2]
            )
        elif self.flip:
            # Non-orientable neighborhood
            self.de_xi_t = torch.clone(self.de_xi_0)

            self.de_xi_t[:, 0] = (
                torch.cos(-self.th_0/2) * self.de_xi_0[:, 0]
                + torch.sin(-self.th_0/2) * self.de_xi_0[:, 2]
            )
            self.de_xi_t[:, 2] = (
                - torch.sin(-self.th_0/2) * self.de_xi_0[:, 0]
                + torch.cos(-self.th_0/2) * self.de_xi_0[:, 2]
            )
            self.de_xi_t[:, 0] = (
                torch.exp(self.mu * self.t) * self.de_xi_0[:, 0]
                )
            self.de_xi_t[:, 2] = (
                torch.exp(self.mu_s * self.t) * self.de_xi_0[:, 2]
                )
            self.de_xi_t[:, 0] = (
                torch.cos(self.th_0/2 + self.om * self.t) * self.de_xi_0[:, 0]
                + torch.sin(self.th_0/2 + self.om * self.t) * self.de_xi_0[:, 2]
            )
            self.de_xi_t[:, 2] = (
                - torch.sin(self.th_0/2 + self.om * self.t) * self.de_xi_0[:, 0]
                + torch.cos(self.th_0/2 + self.om * self.t) * self.de_xi_0[:, 2]
            )
        else: 
            # Expanding-contracting neighborhood
            self.de_xi_t = torch.clone(self.de_xi_0)
            self.de_xi_t[:, 2] = (
                torch.exp(self.mu * self.t) * self.de_xi_0[:, 2]
                )
            self.de_xi_t[:, 0] = (
                torch.exp(self.mu_s * self.t) * self.de_xi_0[:, 0]
                )

        self.xi_t_hat = self.de_xi_t
        self.xi_t_hat[:, 0] += 1.0 

        # evolution in the periodic orbit direction 
        self.th_t = self.th_0 + (self.t / self.T) * 2 * np.pi
        self.xi_t = torch.clone(self.xi_t_hat)
        self.xi_t[:, 0] = (
            torch.cos(self.th_t) * self.xi_t_hat[:, 0] 
            - torch.sin(self.th_t) * self.xi_t_hat[:, 1]
            )
        self.xi_t[:, 1] = (
            torch.sin(self.th_t) * self.xi_t_hat[:, 0] 
            + torch.cos(self.th_t) * self.xi_t_hat[:, 1]
            )
        
        return self.xi_t
    

class ConjNet(pl.LightningModule):
    def __init__(self, encoder, decoder, evolver, lr=1e-3, po=None):
        super().__init__()
        self.encoder = encoder 
        self.decoder = decoder 
        self.evolver = evolver
        self.dim = encoder.dim
        if not(encoder.dim == decoder.dim):
            raise ValueError("encoder.dim and decoder.dim must be equal.")
        self.lr = lr
        
        if type(po) == type(torch.tensor([])):
            self.po = po # first column time stamps
            self.po_constraint = True
        else: 
            self.po_constraint = False
            print("No po data provided, setting po_constraint=False")
            
    def training_step(self, batch, batch_idx):
        txi_0, xi_t = batch

        tet_0 = torch.clone(txi_0[:, 0:4])
        tet_0[:, 0] = txi_0[:, 0]
        xi_0 = txi_0[:, 1:]
        et_0 = self.encoder(xi_0)
        tet_0[:, 1:] = et_0
        et_t = self.evolver(tet_0)
        
        _xi_0 = self.decoder(et_0)
        _xi_t = self.decoder(et_t)

        self.aeloss = F.mse_loss(_xi_0, xi_0)
        self.dynloss = F.mse_loss(_xi_t, xi_t)
        
        if self.po_constraint:
            tt_p = self.po[:, 0]
            et_p = self.encoder(self.po[:, 1:])
            th_p = 2 * torch.pi * tt_p / self.evolver.T
            unit_circ = torch.clone(et_p)
            unit_circ[:, 0] = torch.cos(th_p)
            unit_circ[:, 1] = torch.sin(th_p)
            unit_circ[:, 2] = 0.0
            _xi_p = self.decoder(unit_circ)
            
            self.poloss = (
                F.mse_loss(et_p, unit_circ) 
                + F.mse_loss(_xi_p, self.po[:, 1:])
                )

            loss = self.aeloss + self.dynloss + self.poloss
        else: 
            loss = self.aeloss + self.dynloss
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
            
    def test_step(self, batch, batch_idx):
        txi_0, xi_t = batch

        tet_0 = torch.clone(txi_0[:, 0:4])
        tet_0[:, 0] = txi_0[:, 0]
        xi_0 = txi_0[:, 1:]
        et_0 = self.encoder(xi_0)
        tet_0[:, 1:] = et_0
        et_t = self.evolver(tet_0)
        
        _xi_0 = self.decoder(et_0)
        _xi_t = self.decoder(et_t)

        ae_test_loss = F.mse_loss(_xi_0, xi_0)
        dyn_test_loss = F.mse_loss(_xi_t, xi_t)

        self.log("ae_test_loss", ae_test_loss)
        self.log("dyn_test_loss", dyn_test_loss)

    def validation_step(self, batch, batch_idx):
        
        txi_0, xi_t = batch

        tet_0 = torch.clone(txi_0[:, 0:4])
        tet_0[:, 0] = txi_0[:, 0]
        xi_0 = txi_0[:, 1:]
        et_0 = self.encoder(xi_0)
        tet_0[:, 1:] = et_0
        et_t = self.evolver(tet_0)
        
        _xi_0 = self.decoder(et_0)
        _xi_t = self.decoder(et_t)

        ae_val_loss = F.mse_loss(_xi_0, xi_0)
        dyn_val_loss = F.mse_loss(_xi_t, xi_t)

        self.log("mu", self.evolver.mu)
        self.log("om", self.evolver.om)
        
        self.log("ae_val_loss", ae_val_loss)
        self.log("dyn_val_loss", ae_val_loss)
        
        if self.po_constraint:
            tt_p = self.po[:, 0]
            et_p = self.encoder(self.po[:, 1:])
            th_p = 2 * torch.pi * tt_p / self.evolver.T
            unit_circ = torch.clone(et_p)
            unit_circ[:, 0] = torch.cos(th_p)
            unit_circ[:, 1] = torch.sin(th_p)
            unit_circ[:, 2] = 0.0
            _xi_p = self.decoder(unit_circ)
            
            po_val_loss = (
                F.mse_loss(et_p, unit_circ) 
                + F.mse_loss(_xi_p, self.po[:, 1:])
                )
            val_loss = ae_val_loss + dyn_val_loss + po_val_loss
        else: 
            val_loss = ae_val_loss + dyn_val_loss
        
        self.log("val_loss", val_loss) 
        return val_loss

        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        txi_0 = x 
        tet_0 = torch.clone(txi_0[:, 0:4])
        tet_0[:, 0] = txi_0[:, 0]
        xi_0 = txi_0[:, 1:]
        et_0 = self.encoder(xi_0)
        tet_0[:, 1:] = et_0
        self.et_t = self.evolver(tet_0)
        _xi_t = self.decoder(self.et_t)
        return _xi_t

    def set_lr(self, lr):
        self.lr = lr
