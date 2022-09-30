import os

import torch
from . import SwyftModelForward, SampleStore


class TorchStore:
    """
    A very simple store using torch tensors.
    """

    def __init__(self, store_path: str, simulator: SwyftModelForward):
        self.store_path = store_path
        self.simulator = simulator
        self.samples = None

    def simulate(self, n: int, force_add: bool = False, verbose: bool = True):
        """
        Loads and potentially adds simulations to the store using the
        simulator's prior bounds.

        Args:
            n: number of simulations to return
            force_add: if ``True``, ``n`` new simulations will be added to the
                store, regardless of how many currently lie within bounds
            verbose: if ``True``, prints useful messages

        Returns:
            ``n`` samples from the simulator using its current prior bounds. If
            ``force_add` is false, this will contain simulations in the store
            lying in bounds. If the store does not contain ``n`` simulations in
            bounds, more will be added and saved. If ``force_add`` is false,
            ``n`` new simulations will be added to the store.
        """
        if not os.path.exists(self.store_path):
            if verbose:
                print(f"creating new store with {n} samples")
            self.samples = self.simulator.sample(n)
            torch.save(self.samples, self.store_path)
        else:
            all_samples = torch.load(self.store_path)
            if verbose:
                print(f"loaded store with {len(all_samples)} samples")

            idx = 0  # will be equal to number of samples in bounds after loop
            # Allocate tensor for samples
            self.samples = SampleStore(
                {k: torch.zeros(n, *v.shape[1:]) for k, v in all_samples.items()}
            )
            if not force_add:
                for i in range(len(all_samples)):
                    # If in bounds, copy into samples
                    sample = all_samples[i]
                    if self.simulator.in_bounds(sample):
                        for k in self.samples:
                            self.samples[k][idx] = sample[k]
                        idx += 1
                    if idx == n:
                        break

            if idx < n:
                if verbose:
                    print(f"running {n - idx} new sims")

                # Run new simulations and save
                new_samples = self.simulator.sample(n - idx)

                for k in self.samples:
                    self.samples[k][idx:] = new_samples[k]

                torch.save(
                    SampleStore(
                        {
                            k: torch.cat((v, new_samples[k]))
                            for k, v in all_samples.items()
                        }
                    ),
                    self.store_path,
                )
            elif verbose:
                print("no simulations required")

        return self.samples
