import torch


def double_features(f):
    """Double feature vector as (A, B, C, D) --> (A, A, B, B, C, C, D, D)

    Args:
        f (tensor): Feature vectors (n_batch, n_features)
    Returns:
        f (tensor): Feature vectors (2*n_btach. n_features)
    """
    return torch.repeat_interleave(f, 2, dim=0)


def double_params(params):
    """Double parameters as (A, B, C, D) --> (A, B, A, B, C, D, C, D) etc

    Args:
        params (dict): Dictionary of parameters with shape (n_batch).
    Returns:
        dict: Dictionary of parameters with shape (2*n_batch).
    """
    out = {}
    for k, v in params.items():
        out[k] = torch.repeat_interleave(v.view(-1, 2), 2, dim=0).flatten()
    return out


def loss_fn(head, tail, obs, params):
    # Get features
    f = head(obs)
    n_batch = f.shape[0]

    assert n_batch % 2 == 0, "Loss function can only handle even-numbered batch sizes."

    # Repeat interleave
    f_doubled = double_features(f)
    params_doubled = double_params(params)

    # Get
    lnL = tail(f_doubled, params_doubled)
    lnL = lnL.view(-1, 4, lnL.shape[-1])

    loss = -torch.nn.functional.logsigmoid(lnL[:, 0])
    loss += -torch.nn.functional.logsigmoid(-lnL[:, 1])
    loss += -torch.nn.functional.logsigmoid(-lnL[:, 2])
    loss += -torch.nn.functional.logsigmoid(lnL[:, 3])
    loss = loss.sum(axis=0) / n_batch

    return loss
