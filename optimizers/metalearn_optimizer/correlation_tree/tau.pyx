import numpy as np
cimport numpy as np
from math import sqrt
ctypedef np.int_t int_t


def kendalltau(X, Y):

    """Compute Kendall's tau between two vector X and Y
    
    Parameters
    ----------
    X : 1D array-like container with dtype int
    Y : 1D array-like container dtype int

    The data in X and Y is assumed to be categorical with 
    ordinal scale.


    Returns
    -------
    tau_a : float
        Kendall's tau-a
    tau_b : float
        Kendall's tau-b

    For data in contigency table format, see kendalltau_fromct.

    """ 

    cdef np.ndarray[int_t, ndim=1] _X
    cdef np.ndarray[int_t, ndim=1] _Y
    cdef int P = 0, Q = 0, Tx = 0, Ty = 0
    cdef int n = 0, m, i, j
    cdef int_t *x, *y, x0, y0, x1, y1, q
    
    _X = np.ascontiguousarray(X)
    _Y = np.ascontiguousarray(Y)

    n = <int> _X.shape[0]
    x = <int_t *> _X.data
    y = <int_t *> _Y.data
    
    with nogil:
        for i in range(n):
            x0 = x[i]
            y0 = y[i]
            for j in range(i+1,n):
                x1 = x[j]
                y1 = y[j]
                q = (x1-x0)*(y1-y0)
                if q > 0:
                    # concordant pair
                    P = P + 1
                elif q == 0:
                    # pair with tied rank
                    if (x1 == x0) and (y0 != y1):
                        # tie in x
                        Tx = Tx + 1
                    elif (x1 != x0) and (y0 == y1):
                        # tie in y
                        Ty = Ty + 1
                    # tie in x and y are ignored
                else: 
                    # concordant pair
                    Q = Q + 1

    tau_a = 2*float(P-Q)/(n*(n-1))
    tau_b = float(P-Q)/(sqrt(float((P+Q+Tx)*(P+Q+Ty))))
    return tau_a, tau_b



def kendalltau_fromct(table):

    """Compute Kendall's tau form a contigency table
    
    Parameters
    ----------
    table : 2D array-like container with dtype int
   
    The data is assumed to be contigency table counts 
    with categories on ordinal scale.


    Returns
    -------
    tau_a : float
        Kendall's tau-a
    tau_b : float
        Kendall's tau-b
    tau_c : float
        Kendall's tau-c (also called Stuart's tau-c)

    For computing Kendall's tau between two vectors,
    see kendalltau.

    """ 

    cdef np.ndarray[int_t, ndim=2] _table
    cdef int C, D, P = 0, Q = 0, Tx = 0, Ty = 0
    cdef int i, j, ii, jj, m, n = 0
    cdef int_t *ct, *row, *ct_i, pivot

    _table = np.ascontiguousarray(table)
 
    D = <int> _table.shape[0]
    C = <int> _table.shape[1]
    ct = <int_t *> _table.data
    with nogil:
        for j in range(D):
            for i in range(C):

                pivot = ct[j*C + i] # use this as pivot
                n = n + pivot

                # count concordant pairs
                # -- multiply pivot with 'lower-right' and summate
                for jj in range(j+1,D):
                    row = ct + jj*C
                    for ii in range(i+1,C):
                        P = P + row[ii] * pivot


                # count disconcordant pairs
                # -- multiply pivot with 'lower-left' and summate
                for jj in range(j+1,D):
                    row = ct + jj*C
                    for ii in range(i):
                        Q = Q + row[ii] * pivot

                # count pairs tied in y
                # -- multiply pivot with 'right' and summate
                row = ct + j*C
                for ii in range(i+1,C):
                    Ty = Ty + row[ii] * pivot

                # count pairs tied in x
                # -- multiply pivot with 'below' and summate
                ct_i = ct + i
                for jj in range(j+1,D):
                    row = ct_i + jj*C 
                    Tx = Tx + row[0] * pivot

    tau_a = 2*float(P-Q)/(n*(n-1))
    tau_b = float(P-Q)/(sqrt(float((P+Q+Tx)*(P+Q+Ty))))
    m = C if C < D else D
    tau_c = (P-Q)*(2*m/float((m-1)*n*n))
    return tau_a, tau_b, tau_c



