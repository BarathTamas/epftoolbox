from enum import Enum


class ScalingTypes(Enum):
    """
    Scaling factors types used for X data.
    """
    NORM = "Norm"
    NORM1 = "Norm1"
    STD = "Std"
    MEDIAN = "Median"
    INVARIANT = "Invariant"

    @classmethod
    def has_value(cls, value):
        return value in [v.value for v in cls.__members__.values()]


class FeatureLags(object):
    """
    Class for handling lags defined for the exogenous features.
    The lags can be defined in two different ways:
    1. every feature has the same lags, in this case a list of ints
    is enough;
    2. every feature has different lags, in this case a list of list
    of ints is needed.
    """

    def __init__(self, lags):
        # Checking if lags are validly defined
        assert (isinstance(lags, list))
        self._lags = lags

    def expand_lags(self, n_exog_features):
        """
        If every exog. feature has the same lags, then the list of
        lags needs to be replicated n_exog_features-1 times.

        Parameters
        ----------
        n_exog_features

        Returns
        -------
        None

        """
        shared_lags = self._lags
        self._lags=[shared_lags for i in range(n_exog_features)]
        return

    def to_list(self):
        return self._lags

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Index out of range")
        return self._lags[index]

    def __len__(self):
        return len(self._lags)