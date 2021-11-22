import numpy as np


def estimate_empirical_mass(dataset, post, nobs, npost):
    obs0, _, v0 = dataset[0]
    w0 = post.eval(v0.unsqueeze(0).numpy(), obs0)["weights"]
    mass = {
        k: dict(nominal=[], empirical=np.linspace(1 / nobs, 1, nobs)) for k in w0.keys()
    }
    for _ in range(nobs):
        j = np.random.randint(len(dataset))
        obs0, _, v0 = dataset[j]
        w0 = post.eval(v0.unsqueeze(0).numpy(), obs0)["weights"]
        wS = post.sample(npost, obs0)["weights"]
        for k, v in w0.items():
            f = wS[k][wS[k] >= v].sum() / wS[k].sum()
            mass[k]["nominal"].append(f)
    for k in mass.keys():
        mass[k]["nominal"] = np.asarray(sorted(mass[k]["nominal"]))

    return mass


if __name__ == "__main__":
    pass
