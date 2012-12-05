from __future__ import division, print_function

import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
    double exp(double)
    double log2(double)

dtype = np.float64
itype = np.int

ctypedef np.float64_t dtype_t
ctypedef np.int_t itype_t

cdef long MAXREL_, MID_, MIQ_
MAXREL_ = 0
MID_ = 1
MIQ_ = 2

# make them available from python too
MAXREL = MAXREL_
MID = MID_
MIQ = MIQ_


@cython.boundscheck(False)
cdef dtype_t _mi_h(
        long N,
        itype_t[:] ts,
        itype_t[:] vs,
        itype_t[:] tc,
        itype_t[:] vc,
        long n_tc,
        long n_vc,
        long normalize):

    cdef long i, j, k
    cdef double p_t, p_v, p_tv
    cdef dtype_t mi, h
    cdef long n_t, n_v, n_tv
    cdef itype_t t, v

    if N == 0:
        return 0.0

    mi = 0.0
    h = 0.0

    for i in range(n_tc):
        t = tc[i]
        for j in range(n_vc):
            v = vc[j]

            n_t = 0
            n_v = 0
            n_tv = 0

            for k in range(N):
                if ts[k] == t and vs[k] == v:
                    n_t += 1
                    n_v += 1
                    n_tv += 1
                elif ts[k] == t:
                    n_t += 1
                elif vs[k] == v:
                    n_v += 1

            if n_t != 0 and n_v != 0 and n_tv != 0:
                p_t = n_t / N
                p_v = n_v / N
                p_tv = n_tv / N
                mi += p_tv * log2(p_tv / (p_t * p_v))
                if normalize:
                    h += -p_tv * log2(p_tv)

    if normalize:
        # add 0.0001 to prevent division by zero
        return mi / (h + 0.0001)
    else:
        return mi

@cython.boundscheck(False)
def _mrmr(
        long N,
        long M,
        itype_t[:] ts,
        itype_t[:, :] vs,
        itype_t[:] tc,
        itype_t[:] vc,
        long n_tc,
        long n_vc,
        long K,
        long method,
        long normalize):

    cdef long i, j, k
    cdef np.ndarray[dtype_t, ndim=1] relevances
    cdef np.ndarray[dtype_t, ndim=1] redundancies
    cdef np.ndarray[itype_t, ndim=1] ks
    cdef np.ndarray[dtype_t, ndim=1] scores
    cdef double max_score, score
    cdef long idx_, skip

    i = 0
    j = 0
    k = 0

    relevances = np.zeros(M, dtype=dtype)
    redundancies = np.zeros(M, dtype=dtype)
    ks = np.zeros(K, dtype=itype)
    scores = np.zeros(K, dtype=dtype)

    # precompute mutual info with target variable
    for i in range(M):
        relevances[i] = _mi_h(N, ts, vs[:, i], tc, vc, n_tc, n_vc, normalize)

    # perform feature selection
    for i in range(K):
        max_score = 0.0
        idx_ = 0

        for j in range(M):
            # skip selected variables
            if i > 0:
                skip = False
                for k in range(i):
                    if ks[k] == j:
                        skip = True
                if skip:
                    continue

            # accumulate the mutual info with previously selected variables
            if i > 0 and method != MAXREL_:
                redundancies[j] += _mi_h(N, vs[:, ks[i - 1]], vs[:, j], vc, vc, n_vc, n_vc, normalize)

            # if we're maxrel or we're on our first feature
            if i == 0 or method == MAXREL_:
                score = relevances[j]
            elif method == MID_:
                score = relevances[j] - (redundancies[j] / i)
            else:
                # add 0.0001 to prevent division by zero error
                score = relevances[j] / ((redundancies[j] / i) + 0.0001)

            if score > max_score:
                idx_ = j
                max_score = score

        ks[i] = idx_
        scores[i] = max_score

    return ks, scores
