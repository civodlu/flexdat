from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from mypy_extensions import NamedArg

Pairing = List[Sequence[int]]  # list of associated ID
PairingSampler = Callable[
    [
        NamedArg(int, 'base_index'),
        NamedArg(Pairing, 'pairing'),
        NamedArg(Optional[Dict], 'context'),
        NamedArg(Optional[Dict[str, Sequence[Any]]], 'pairing_metadata'),
    ],
    Sequence[Tuple[str, int],],
]  # return list (prefix, index)


class PairingSamplerRandom:
    """
    Return pairs, triplet, ... randomly selected from the pairing
    """

    def __init__(
        self,
        nb_pairings: int = 2,
        pairing_key_prefix: Optional[Sequence[str]] = None,
        with_replacement: bool = False,
        always_select_first_sample: bool = True,
    ) -> None:
        """
        Args:
            nb_pairings: the size of the pairing (2 = pair, 3 = triplet, ...)
            always_select_first_sample: if True, the first sample from the association will always be selected as
                first (e.g., if we have different image qualities for denoising we want to consider the first
                image as the full)
            with_replacement: if True, an index in a pairing can be drawn only once
            pairing_key_prefix: appropriate naming for the pairing. If not defined, `p{n}_` prefix will be used
        """
        self.nb_pairings = nb_pairings
        self.with_replacement = with_replacement
        self.always_select_first_sample = always_select_first_sample

        if pairing_key_prefix is not None:
            assert len(pairing_key_prefix) == nb_pairings, f'each sample of a pairing must be named or none. Expected={nb_pairings}, got={len(pairing_key_prefix)}'
            self.pairing_key_prefix = pairing_key_prefix
        else:
            self.pairing_key_prefix = [f'p{n}_' for n in range(nb_pairings)]

    def __call__(
        self,
        base_index: int,
        pairing: Pairing,
        context: Optional[Dict],
        pairing_metadata: Optional[Dict[str, Sequence[Any]]],
    ) -> Sequence[Tuple[str, int]]:

        choices = pairing[base_index]
        if self.always_select_first_sample:
            i = choices[0]
            results = [(self.pairing_key_prefix[0], i)]

            other_i = np.random.choice(choices[1:], self.nb_pairings - 1, replace=self.with_replacement)
            results += list(zip(self.pairing_key_prefix[1:], other_i))
        else:
            other_i = np.random.choice(choices, self.nb_pairings, replace=self.with_replacement)
            results = list(zip(self.pairing_key_prefix, other_i))

        return results


class PairingSamplerEnumerate:
    """
    Sampler that enumerates in order the pairing
    """

    def __init__(self, prefix: str = 'sample_'):
        self.prefix = prefix

    def __call__(
        self,
        base_index: int,
        pairing: Pairing,
        context: Optional[Dict],
        pairing_metadata: Optional[Dict[str, Sequence[Any]]],
    ) -> Sequence[Tuple[str, int]]:

        choices = pairing[base_index]
        pairing_sampled = [(f'{self.prefix}{c_n}_', c) for c_n, c in enumerate(choices)]
        return pairing_sampled


class PairingSamplerEnumerateNamed:
    """
    Sampler that enumerates in order the pairing and give them a fixed name
    """
    def __init__(self, names: Sequence[str]):
        self.names = names

    def __call__(
        self,
        base_index: int,
        pairing: Pairing,
        context: Optional[Dict],
        pairing_metadata: Optional[Dict[str, Sequence[Any]]],
    ) -> Sequence[Tuple[str, int]]:

        choices = pairing[base_index]
        assert len(choices) == len(self.names)
        pairing_sampled = [(f'{c_n}', c) for c_n, c in zip(self.names, choices)]
        return pairing_sampled