
def mrmr_method(string):
    method = string.upper()
    if method == 'MAXREL':
        return MAXREL
    elif method == 'MID':
        return MID
    elif method == 'MIQ':
        return MIQ
    else:
        raise ValueError("MRMR method must be one of 'MAXREL', 'MID', or 'MIQ'")

def test1(args=None):
    import csv
    import os.path
    import sys
    import numpy as np
    from argparse import ArgumentParser, FileType
    from time import time
    import sklmrmr
    from sklmrmr._mrmr import _mrmr as mrmr, MAXREL, MID, MIQ

    default_file = os.path.join(
        os.path.dirname(sklmrmr.__file__),
        'test_nci9_s3.csv'
        )

    if args is None:
        args = sys.argv[1:]

    parser = ArgumentParser(description='minimum redundancy maximum relevance feature selection')
    parser.add_argument('--method', type=mrmr_method, default=MID)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--n_features', type=int, default=10)
    parser.add_argument('--file', type=FileType('r'), default=open(default_file))
    ns = parser.parse_args(args)

    rdr = csv.reader(ns.file)
    raw_data = [row for row in rdr]

    if ns.file != sys.stdin:
        ns.file.close()

    N = len(raw_data) - 1
    M = len(raw_data[0]) - 1

    ts = np.zeros((N,), dtype=int)
    vs = np.zeros((N, M), dtype=int)

    for i, row in enumerate(raw_data[1:]):
        ts[i] = int(row[0])
        for j, v in enumerate(row[1:]):
            vs[i, j] = int(v)

    tc = np.array(list(set(ts)))
    vc = np.array(list(set(vs.reshape((N * M,)))))

    n_tc = len(tc)
    n_vc = len(vc)

    t = time()

    ks, scores = mrmr(N, M, ts, vs, tc, vc, n_tc, n_vc, ns.n_features,
            ns.method, ns.normalize)

    t = time() - t

    print("took %.3fs\n" % t)

    print("*** %s features ***" % ("MaxRel" if ns.method == MAXREL else "mRMR"))
    for i in range(ns.n_features):
        idx = ks[i] + 1
        print("%d\t%s\t%.3f" % (idx, raw_data[0][idx], scores[i]))

    return 0

def test2():
    from sklearn.svm import SVC
    from sklearn.datasets import load_digits
    from sklmrmr import MRMR

    digits = load_digits()
    X = digits.images.reshape((len(digits.images), -1)).astype(int)
    y = digits.target

    svc = SVC(kernel='linear', C=1)
    mrmr = MRMR(estimator=svc, n_features_to_select=5)
    mrmr.fit(X, y)
    ranking = mrmr.ranking_

    print(ranking)

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(test1())
