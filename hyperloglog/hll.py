"""
This module implements probabilistic data structure which is able to calculate the cardinality of large multisets in a single pass using little auxiliary memory
"""

from cStringIO import StringIO
from hashlib import sha1
import math
import struct
import zlib

import numpy
# from scipy.stats.mstats import hmean

from const import rawEstimateData, biasData, thresholdData

EXACT = 1
ARRMODE = 2

class HyperLogLog(object):
  """
  HyperLogLog cardinality counter
  """

  __slots__ = ('error_rate', 'alpha', 'p', 'm', 'M', 'exact_thresh', 'exact_set', 'mode', 'added_count')
  _magic_slots = ('M', '_fmt', 'exact_set', '_exact_size')

  def __init__(self, error_rate):
    """
    Implementes a HyperLogLog

    error_rate = abs_err / cardinality
    """

    if not (0 < error_rate < 1):
      raise ValueError("Error_Rate must be between 0 and 1.")

    # error_rate = 1.04 / sqrt(m)
    # m = 2 ** p
    # M(1)... M(m) = 0

    p = int(math.ceil(math.log((1.04 / error_rate) ** 2, 2)))

    self.error_rate = error_rate
    self.alpha = self._get_alpha(p)
    self.p = p
    self.m = 1 << p
    self.exact_thresh = self.m // 8
    self.M = None
    self.exact_set = set()
    self.mode = EXACT
    self.added_count = 0

  @staticmethod
  def get_hash(value):
    return struct.unpack('>Qxxxxxxxxxxxx', sha1(value).digest())[0]

  def add(self, value):
    """
    Adds the item to the HyperLogLog
    """
    if value is None:
      return

    if not isinstance(value, basestring):
      value = str(hash(value)) * 10

    self.addhash(self.get_hash(value))

  def addhash(self, hsh):

    if self.mode == EXACT:
      self.exact_set.add(hsh)

      if self.ready_to_upgrade():
        self.upgrade()

    else:
      self.update_arr(hsh)

    self.added_count += 1

  def extend(self, iterable):
    for i in iterable:
      self.add(i)

  def ready_to_upgrade(self):
    return (
        self.exact_set is not None
        and len(self.exact_set) > self.exact_thresh)

  def upgrade(self):
    if self.mode == EXACT:
      self.M = numpy.zeros(self.m)
      for hsh in self.exact_set:
        self.update_arr(hsh)
      self.exact_set = None
      self.mode = ARRMODE

  def update_arr(self, hsh):
    j, w = self.munge_hash(hsh)
    self.M[j] = max(self.M[j], w)

  def munge_hash(self, value):
    j = value & (self.m - 1)
    w = value >> self.p

    return (j, self._get_rho(w, 64 - self.p))

  def update(self, *others):
    """
    Merge other counters
    """

    for other in others:
      if other.mode == EXACT:
        for item in other.exact_set:
          self.addhash(item)

      elif other.mode == ARRMODE:
        if self.m != item.m:
          raise ValueError('Counters precisions should be equal')

        elif self.mode != ARRMODE:
          self.upgrade()

        self.M = numpy.maximum(self.M, other.M)
        self.added_count += other.added_count

  def __eq__(self, other):
    if self.added_count == other.added_count == 0:
      return True

    elif self.mode == other.mode == EXACT:
      return self.exact_set == other.exact_set

    elif self.mode == other.mode == ARRMODE:
      return (self.M == other.M).all()

    elif self.mode == EXACT and other.mode == ARRMODE:
      cpy = self.copy()
      cpy.upgrade()
      return (cpy.M == other.M).all()

    elif self.mode == ARRMODE and other.mode == EXACT:
      cpy = other.copy()
      cpy.upgrade()
      return (self.M == cpy.M).all()

  def __ne__(self, other):
    return not self.__eq__(other)

  def __add__(self, other):
    newone = self.__class__()
    newone.update(self, others)

  def __iadd__(self, other):
    self.update(self, other)

  def __empty__(self):
    return not self.added_count

  def __len__(self):
    if self.added_count > 0:
      return round(self.card())
    else:
      return 0

  def __gt__(self, other):
    return self.card() > other.card()

  def __lt__(self, other):
    return self.card() < other.card()

  def _Ep(self):
    estimate = self.alpha * self.m ** 2 / (2 ** -self.M).sum()
    # estimate = self.alpha * self.m * hmean(self.M)
    if estimate <= 5 * self.m:
      return (estimate - self._get_bias_estimate(estimate, self.p))
    else:
      return estimate

  def card(self):
    """
    Returns the estimate of the cardinality
    """

    if self.added_count == 0:
      return 0

    elif self.mode == EXACT:
      return len(self.exact_set)

    elif not self.M.all():
      V = self.m - numpy.count_nonzero(self.M)
      H = self.m * math.log(self.m / float(V))
      if H <= self._get_bias_threshold(self.p):
        return H
      else:
        return self._Ep()
    else:
      return self._Ep()

  def copy(self):
    cpy = self.__class__(self.error_rate)
    for k in self.__slots__:
      if k not in self._magic_slots:
        setattr(cpy, getattr(self, k))

    if self.mode == EXACT:
      cpy.exact_set = self.exact_set.copy()
    else:
      cpy.M = self.M.copy()

    return cpy

  def serialize(self):
    d = {}
    for val in self.__slots__:
      if val not in self._magic_slots:
        d[val] = getattr(self, val)

    if self.exact_set:
      d['_exact_size'] = len(self.exact_set)
      d['exact_set'] = zlib.compress(struct.pack('>' + 'Q' * len(self.exact_set), *sorted(self.exact_set)))

    if self.M is not None and self.M.any():
      maxval = self.M.max()
      fmt = self._get_fmt(maxval)
      d['_fmt'] = fmt
      d['M'] = zlib.compress(struct.pack('>' + fmt * self.m, *self.M))

    return d

  @staticmethod
  def _get_fmt(maxval):
    if maxval is None or maxval < 0:
      raise ValueError
    elif maxval <= 0xff:
      return 'B'
    elif maxval <= 0xffff:
      return 'H'
    elif maxval <= 0xffffffff:
      return 'I'
    elif maxval <= 0xffffffffffffffff:
      return 'L'
    elif maxval <= 0xffffffffffffffffffffffffffffffff:
      return 'Q'
    else:
      raise ValueError

  @classmethod
  def deserialize(cls, d):
    self = cls(d['error_rate'])
    diter = cls._deserialize(d)
    for k, v in diter:
      setattr(self, k, v)

    return self

  @classmethod
  def _deserialize(cls, d):
    for k, v in d.iteritems():
      if k not in cls._magic_slots:
        yield k, v

    if 'M' in d:
      yield 'M', numpy.array(
        struct.unpack('>' + d['_fmt'] * d['m'], zlib.decompress(d['M'])))

    if 'exact_set' in d:
      yield 'exact_set', set(
        struct.unpack('>' + 'Q' * d['_exact_size'], zlib.decompress(d['exact_set'])))

  def __getstate__(self):
    return self.serialize()

  def __setstate__(self, d):
    diter = self._deserialize(d)
    for k, v in diter:
      setattr(self, k, v)

  @staticmethod
  def _get_bias_threshold(p):
    return thresholdData[p - 4]

  @staticmethod
  def _get_bias_estimate(E, p):
    bias_vector = biasData[p - 4]
    estimate_vector = rawEstimateData[p - 4]
    nearest_neighbors = ((estimate_vector - E) ** 2).argsort()[:6]
    return bias_vector[nearest_neighbors].mean()

  @staticmethod
  def _get_alpha(p):
    if not (4 <= p <= 16):
      raise ValueError("p=%d should be in range [4 : 16]" % p)

    if p == 4:
      return 0.673

    if p == 5:
      return 0.697

    if p == 6:
      return 0.709

    return 0.7213 / (1.0 + 1.079 / (1 << p))

  @staticmethod
  def _get_bit_length(i):
    if not i:
      return 0
    else:
      return len(bin(i)) - 2

  @classmethod
  def _get_rho(cls, w, max_width):
    rho = max_width - cls._get_bit_length(w) + 1

    if rho <= 0:
      raise ValueError('w overflow')

    return rho
