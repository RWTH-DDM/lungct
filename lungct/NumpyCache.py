
import hashlib
import numpy as np
import os
from typing import Callable


class NumpyCache:

    def __init__(self, path):

        self._path = path
        self._memory_cache = {}

    def get_cached(self, fn: Callable[..., np.array], cache_key: str, *arguments):

        """ Returns cached or computed result of the given function. """

        hash_context = hashlib.md5()
        hash_context.update(fn.__name__.encode('utf-8'))
        hash_context.update(cache_key.encode('utf-8'))
        cache_key = hash_context.hexdigest()

        # data not in memory
        if cache_key not in self._memory_cache:

            cache_file_path = os.path.join(self._path, cache_key + '.npy')

            # try to load from filesystem cache
            if os.path.isfile(cache_file_path):
                self._memory_cache[cache_key] = np.load(cache_file_path)

            # compute function result
            else:
                self._memory_cache[cache_key] = fn(*arguments)
                np.save(cache_file_path, self._memory_cache[cache_key])

        return self._memory_cache[cache_key]
