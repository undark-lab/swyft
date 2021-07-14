import torch


def double_features(f: torch.Tensor) -> torch.Tensor:
    """Double feature vector as (A, B, C, D) --> (A, A, B, B, C, C, D, D)

    Args:
        f: Feature vectors (n_batch, n_features)
    Returns:
        f: Feature vectors (2*n_btach. n_features)
    """
    return torch.repeat_interleave(f, 2, dim=0)


def double_params(params: torch.Tensor) -> torch.Tensor:
    """Double parameters as (A, B, C, D) --> (A, B, A, B, C, D, C, D) etc

    Args:
        params: Dictionary of parameters with shape (n_batch).
    Returns:
        dict: Dictionary of parameters with shape (2*n_batch).
    """
    n = params.shape[-1]
    out = torch.repeat_interleave(params.view(-1, 2 * n), 2, dim=0).view(-1, n)
    return out


def loss_fn(
    head: torch.nn.Module,
    tail: torch.nn.Module,
    obs: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
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
