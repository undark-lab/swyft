#############
# OLD OLD OLD
#############

def get_posteriors(network, x0, z, error = False, n_sub = None):
    """Get posteriors, potentially with MC dropout uncertainties.
    """
    zsub = subsample(n_sub, z)
    if not error:
        network.eval()
        z_lnL = estimate_lnL(network, x0, zsub, normalize = True)
        return z_lnL
    else:
        network.train()
        z_lnL_list = []
        for i in tqdm(range(100), desc="Estimating std"):
            z_lnL = estimate_lnL(network, x0, zsub, normalize = False)
            z_lnL_list.append(z_lnL)
        std_list = []
        for j in range(len(z_lnL_list[0])):
            tmp = [z_lnL_list[i][j]['lnL'] for i in range(len(zsub))]
            mean = sum(tmp)/len(zsub)
            tmp = [(z_lnL_list[i][j]['lnL']-mean)**2 for i in range(len(zsub))]
            var = sum(tmp)/len(zsub)
            std = var**0.5
            std_list.append(std)
        return std_list

def resample_z(n, z_seeds, epsilon = None):
    z_samples = []
    n_dim = len(z_seeds)
    for i in range(n_dim):
        z = z_seeds[i].reshape(-1, 1)
        tree = BallTree(z)
        # Estimate epsilon as  epsilon = 4 * (average nn-distance)
        if epsilon is None:
          nn_dist = tree.query(z, 2)[0][:,1]
          epsilon = nn_dist.mean() * 4.
        z_new = []
        counter = 0
        while counter < n:
            z_proposal = torch.rand(n, 1).to(z_seeds[0].device);
            nn_dist = tree.query(z_proposal, 1)[0][:,0]
            mask = nn_dist <= epsilon
            z_new.append(z_proposal[mask])
            counter += mask.sum()
        z_new = torch.cat(z_new)[:n]
        z_samples.append(z_new)
    z = [torch.cat([z_samples[i][j] for i in range(n_dim)]) for j in range(n)]
    return z

def init_xz(model, n_sims, n_dim):
    z = sample_z(n_sims, n_dim)
    xz = sample_x(model, z)
    return xz

def update_xz(xz, network, x0, model, n_sims, lnL_th = -6, n_sub = 1000, append = True):
    # Generate training points
    z_sub = subsample(n_sub, get_z(xz))
    z_lnL = estimate_lnL(network, x0, z_sub)
    z_seeds = get_seeds(z_lnL, lnL_th = lnL_th)
    z_new = resample_z(n_sims, z_seeds)

    # Generate training data
    xz_new = sample_x(model, z_new)

    # Append or not append
    if append:
        xz += xz_new
    else:
        xz = xz_new
    return xz

