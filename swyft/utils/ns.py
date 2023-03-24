import torch
import numpy as np
from tqdm.auto import tqdm

class SwyftSimpleSliceSampler:
    def __init__(self, X_init):
        self.X_init = X_init
        self.X_live = X_init*1.
        self.L_live = None
        self.device = X_init.device
        self._update_L(X_init)

    def _get_directions(self, B, D):
        """This function generates a minibatch of D-dimensional random directions."""
        t = torch.randn(B, D, device = self.device)
        l = (t**2).sum(axis=-1)**0.5
        n = t/l.unsqueeze(-1)
        return n

    def _get_slice_sample_points(self, B, S):
        """This function generates a minibatch of S slice sample positions."""
        hard_bounds = torch.tensor([-1., 1.], device = self.device).unsqueeze(0).repeat((B,1))
        current_bounds = hard_bounds.clone()
        L = torch.empty((B, S), device = self.device)
        for i in range(S):
            x = torch.rand(B, device = self.device)*(current_bounds[:,1]-current_bounds[:,0])+current_bounds[:,0]
            L[:,i] = x
            current_bounds[x<0,0] = x[x<0]
            current_bounds[x>0,1] = x[x>0]
        return L
    
    def _gen_new_samples(self, X_seeds, logl_fn, logl_th, num_steps = 5, max_step_size = 1., samples_per_slice = 20):
        """This function generates new samples within the likelihodo constraint logl_fn > log_th."""
        B, D = X_seeds.shape
        C = torch.zeros(B, device = self.device)  # counter for accepted points
        X = X_seeds.clone()
        logl = torch.ones(B, device = self.device)*(-np.inf)
        for i in range(num_steps):
            N = self._get_directions(B, D)
            L = self._get_slice_sample_points(B, S = samples_per_slice)*max_step_size
            dX_uniform = N.unsqueeze(-2)*L.unsqueeze(-1)
            dX = torch.matmul(dX_uniform, self._L.T)
            pX = X.unsqueeze(-2) + dX # proposals
            #logl_prop = logl_fn(pX.flatten(0, -2)).view(*pX_shape[:-2], -1)
            logl_prop = logl_fn(pX)
            accept_matrix = logl_prop > logl_th
            #print(accept_matrix.sum(), logl_th)
            idx = torch.argmax(accept_matrix.int(), dim=1)
            nX = torch.stack([pX[i][idx[i]] for i in range(B)], dim=0)
            logl_selected = torch.stack([logl_prop[i][idx[i]] for i in range(B)], dim=0)
            accept_any = (accept_matrix.sum(dim=-1)>0)
            X[accept_any] = nX[accept_any]
            logl[accept_any] = logl_selected[accept_any]
            C[accept_any] += 1
        return X[C==num_steps], logl[C==num_steps]
    
    def nested_sampling(self, logl_fn, logl_th_max = np.inf, max_steps = 100000, num_batch_samples = 100, epsilon = 1e-6):
        """Run nested sampling, staring with X_init live points."""
        X_init = self.X_live
        NLP, D = X_init.shape
        X_live = X_init.clone()
        if self.L_live is None:
            L_live = logl_fn(X_live)
        else:
            L_live = self.L_live
        B = min(num_batch_samples, NLP)  # Number of samples generated simultanously
        V = 1.  # initial volume is set to 1.
        samples_X = []
        samples_logl = []  # logl values
        samples_logv = []  # estimate of constrained volume
        samples_Z = []
        samples_logwt = []
        logl_th = torch.tensor(-np.inf)
        Z = 0
        Z_rest = torch.tensor(np.inf)

        pbar = tqdm(range(max_steps))
        for i in pbar:
            pbar.set_description("Z_sum=%.2e, Z_rest=%.2e, logl_min=%.2f"%(Z, Z_rest.item(), logl_th.item()))
   #        pbar.set_description("logl_min=%.2f, Z=%.2e"%(logl_th, Z))
   #         pbar.set_description("Z_sum=%.2e, Z_rest=%.2e, logl_min=%.2f"%(Z, Z_rest, logl_th))
            self._update_L(X_live)
            idx_batch = np.random.choice(range(NLP), B, replace = True)
            X_batch = X_live[idx_batch]
            logl_th = L_live.min()
            if logl_th > logl_th_max:  # Stop sampling once maxmimum threshold is reached
                break
            X_new, L_new = self._gen_new_samples(X_batch, logl_fn, logl_th, num_steps = 10)
            #print(len(X_new), X_live.min(), X_live.max(), L_live.min(), L_live.max())
            for i in range(len(X_new)):
                if L_new[i] > L_live.min():
                    idx_min = np.argmin(L_live.cpu().numpy())
                    Lmin = L_live[idx_min].item()*1.
                    samples_X.append(1.*X_live[idx_min].cpu().numpy())
                    samples_logl.append(Lmin)
                    samples_logv.append(np.log(V))
                    L_live[idx_min] = L_new[i]*1.0
                    X_live[idx_min] = X_new[i]*1.0
                    V *= (1-1/NLP)  # Volume estimate per sample
                    dZ = V/NLP*np.exp(Lmin)
                    samples_Z.append(dZ)
                    samples_logwt.append(Lmin + np.log(V/NLP))
                    Z = Z + dZ
                    Z_rest = V*torch.exp(L_live.max()*1.)
                    #print(L_live.max()*1.)
                else:
                    break
            if Z_rest < Z*epsilon:
    #            print(Z_rest, Z)
                break
        samples_logv = torch.tensor(np.array(samples_logv)).float()
        samples_logl = torch.tensor(np.array(samples_logl)).float()
        samples_X = torch.tensor(np.array(samples_X)).float()
        samples_logwt = torch.tensor(np.array(samples_logwt)).float()
        self.X_live = X_live
        self.L_live = L_live
        self.samples_X = samples_X
        self.samples_logv = samples_logv
        self.samples_logl = samples_logl
        self.samples_logwt = samples_logwt

    def grow_live_points(self, logl_fn, logl_th_max = -1.0, num_batch_samples = 100, num_samples = 10000):
        X_init = self.X_live
        NLP, D = X_init.shape
        X_live = X_init.clone()
        L_live = logl_fn(X_live)
        B = min(num_batch_samples, NLP)  # Number of samples generated simultanously
        while len(X_live) < num_samples:
            NLP = len(X_live)
            idx_batch = np.random.choice(range(NLP), B, replace = True)
            X_batch = X_live[idx_batch]
            X_new, L_new = self._gen_new_samples(X_batch, logl, logl_th_max, num_steps = 10)
            X_live = torch.cat([X_live, X_new])
            L_live = torch.cat([L_live, L_new])
        return X_live, L_live
    
    def get_threshold(self, p):
    #    logwt = samples_logl[1:]+np.log(samples_logv[:-1]-samples_logv[1:])
        wt = np.exp(self.samples_logwt)
        wt /= wt.sum()
        cwt = np.cumsum(wt)
        return np.interp(p, cwt, self.samples_logl)

    def get_posterior_samples(self, N = None):
        if N is None:
            N = int(self.get_posterior_neff())
        logwt = self.samples_logwt
        wt = torch.exp(logwt - logwt.max())
        wt = torch.tensor(wt)
        wt /= wt.sum()
        idx = torch.multinomial(wt, N, replacement = True)
        return self.samples_X[idx], self.samples_logl[idx]

    def get_posterior_neff(self):
        logwt = self.samples_logwt
        wt = torch.exp(logwt - logwt.max())
        wt = torch.tensor(wt)
        wt /= wt.sum()
        n_eff = sum(wt)**2/sum(wt**2)
        return n_eff.item()
    
    def get_constrained_prior_samples(self, N = None, min_logl = -np.inf):
        if N is None:
            N = int(self.get_constrained_prior_neff(min_logl))
        logv = self.samples_logv
        logl = self.samples_logl
        mask = logl >= min_logl
        wt = torch.exp(logv[mask]-logv[mask].max())
        idx = torch.multinomial(wt, N, replacement = True)
        return self.samples_X[mask][idx], self.samples_logl[mask][idx]
    
    def get_constrained_prior_neff(self, min_logl):
        logv = self.samples_logv
        logl = self.samples_logl
        mask = logl >= min_logl
        wt = torch.exp(logv[mask]-logv[mask].max())
        n_eff = sum(wt)**2/sum(wt**2)
        return n_eff.item()
    
    def _update_L(self, X):
        #X = self.X_live*1.
        cov = torch.cov(X.T)
        L = torch.linalg.cholesky(cov)
        self._L = L

