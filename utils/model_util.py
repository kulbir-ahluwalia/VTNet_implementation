""" Borrowed from https://github.com/andrewliao11/pytorch-a3c-mujoco/blob/master/model.py."""

class ScalarMeanTracker(object):
    def __init__(self) -> None:
        self._sums = {}
        self._counts = {}

    def add_scalars(self, scalars):
        for k in scalars:
            if k != 'tools':
                if k not in self._sums:
                    self._sums[k] = scalars[k]
                    self._counts[k] = 1
                else:
                    self._sums[k] += scalars[k]
                    self._counts[k] += 1

    def pop_and_reset(self):
        means = {k: self._sums[k] / self._counts[k] for k in self._sums}
        self._sums = {}
        self._counts = {}
        return means
