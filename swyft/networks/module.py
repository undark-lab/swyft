import logging

import torch.nn as nn


class Module(nn.Module):
    """Thin wrapper around pytorch modules that enables automatic reloading of network classes with correct arguments."""

    registry = {}

    def __init__(self, *args, **kwargs):
        """Store arguments of subclass instantiation."""
        self._swyft_args = [args, kwargs]
        super().__init__()
        logging.debug("Initializing swyft.Module with tag `%s`" % self._swyft_tag)
        logging.debug("  args = `%s`" % str(args))
        logging.debug("  kwargs = `%s`" % str(kwargs))

    def __init_subclass__(cls, **kwargs):
        """Register subclasses."""
        super().__init_subclass__(**kwargs)
        cls.registry[cls.__name__] = cls
        cls._swyft_tag = cls.__name__
        logging.debug("Registering new swyft.Module with tag `%s`" % cls._swyft_tag)

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
