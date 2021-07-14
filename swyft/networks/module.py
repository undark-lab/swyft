import logging

import torch.nn as nn

log = logging.getLogger(__name__)


class Module(nn.Module):
    """swyft.Module, a wrapper around torch.nn.Module which enables to store
    and load swyft posteriors with custom networks.

    .. note::
        This is a thin wrapper around pytorch modules that enables automatic
        reloading of network classes with correct arguments.

    .. warning::
        Please use `swyft.Module` rather then `torch.nn.Module` when defining
        custom tail or head networks. Otherwise posteriors cannot be properly
        saved.  Note thta uou MUST call the super-class `__init__` function
        with all arguments, which will take care of storage.

    Example::
        >>> class MyHead(swyft.Module):
        >>>     def __init__(self, sim_shapes, custom_parameter):
        >>>         super().__init__(sim_shapes = sim_shapes, custom_parameters = cumstom_parameters)
        >>>         # ...
    """

    registry = {}

    def __init__(self, *args, **kwargs):
        """Store arguments of subclass instantiation."""
        self._swyft_args = [args, kwargs]
        super().__init__()
        log.debug("Initializing swyft.Module with tag `%s`" % self._swyft_tag)
        log.debug("  args = `%s`" % str(args))
        log.debug("  kwargs = `%s`" % str(kwargs))

    def __init_subclass__(cls, **kwargs):
        """Register subclasses."""
        super().__init_subclass__(**kwargs)
        cls.registry[cls.__name__] = cls
        cls._swyft_tag = cls.__name__
        log.debug("Registering new swyft.Module with tag `%s`" % cls._swyft_tag)

    @property
    def swyft_args(self):
        return self._swyft_args

    @property
    def swyft_tag(self):
        """Return subclass tag."""
        # TODO: Confirm this is stable. Alternative is to as users to provide tags.
        return self._swyft_tag

    def swyft_state_dict(self):
        torch_state_dict = self.state_dict()
        return dict(
            torch_state_dict=torch_state_dict,
            swyft_args=self.swyft_args,
            swyft_tag=self.swyft_tag,
        )

    @classmethod
    def from_swyft_state_dict(cls, state_dict):
        subcls = cls.registry[state_dict["swyft_tag"]]
        args, kwargs = state_dict["swyft_args"]
        instance = subcls(*args, **kwargs)
        instance.load_state_dict(state_dict["torch_state_dict"])
        return instance
