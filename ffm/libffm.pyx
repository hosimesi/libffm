# cython: language_level=3
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdlib cimport free
cimport numpy as cnp

cnp.import_array()


cdef extern from "ffm.h" namespace "ffm" nogil:
    ctypedef float ffm_float
    ctypedef double ffm_double
    ctypedef int ffm_int
    ctypedef long long ffm_long

    struct ffm_node:
        ffm_int f
        ffm_int j
        ffm_float v

    struct ffm_problem:
        ffm_int n
        ffm_int l
        ffm_int m

        ffm_node* X
        ffm_long* P
        ffm_float* Y

    struct ffm_importance_weights:
        ffm_int l
        ffm_float sum
        ffm_float *W

    struct ffm_parameter:
        ffm_float eta
        ffm_float lambda_ "lambda"
        ffm_int nr_iters
        ffm_int k
        ffm_int nr_threads
        ffm_int auto_stop_threshold
        char *json_meta_path
        bint quiet
        bint normalization
        bint random
        bint auto_stop

    struct ffm_model:
        ffm_int n
        ffm_int m
        ffm_int k
        ffm_float *W
        bint normalization
        ffm_int best_iteration

    ffm_model *ffm_train_with_validation(ffm_problem *Tr, ffm_problem *Va, ffm_importance_weights *iws, ffm_importance_weights *iwvs, ffm_parameter param);


cdef ffm_problem* make_ffm_prob(X, y):
    if len(X) != len(y):
        raise ValueError("X and y should contain the same length items")

    cdef:
        ffm_int l = len(X)
        ffm_long nnz = sum(len(x) for x in X)
        ffm_long p = 0

    cdef ffm_problem* prob = <ffm_problem *> PyMem_Malloc(sizeof(ffm_problem))
    if prob is NULL:
        raise MemoryError("Insufficient memory for prob")
    cdef ffm_node* nodes = <ffm_node *> PyMem_Malloc(nnz * sizeof(ffm_node))
    if nodes is NULL:
        raise MemoryError("Insufficient memory for data")
    cdef ffm_long* position = <ffm_long *> PyMem_Malloc((l+1) * sizeof(ffm_long))
    if position is NULL:
        raise MemoryError("Insufficient memory for positions")
    cdef ffm_float* labels = <ffm_float *> PyMem_Malloc(l * sizeof(ffm_float))
    if labels is NULL:
        raise MemoryError("Insufficient memory for labels")

    m, n = 0, 0
    position[0] = 0
    for i, (x_, y_) in enumerate(zip(X, y)):
        labels[i] = 1.0 if y_ > 0 else -1.0
        for field, column, val in x_:
            m = max(m, field + 1)
            n = max(n, column + 1)
            nodes[p].f = field
            nodes[p].j = column
            nodes[p].v = val
            p += 1
        position[i + 1] = p

    prob.n = n
    prob.m = m
    prob.l = l
    prob.X = nodes
    prob.Y = labels
    prob.P = position
    return prob


cdef void free_ffm_prob(ffm_problem* prob):
    if prob is NULL:
        return
    if prob.X is not NULL:
        PyMem_Free(prob.X)
    if prob.Y is not NULL:
        PyMem_Free(prob.Y)
    if prob.P is not NULL:
        PyMem_Free(prob.P)
    PyMem_Free(prob)


cdef ffm_importance_weights* make_ffm_iw(weights):
    cdef:
        ffm_float* W = <ffm_float *> PyMem_Malloc(len(weights) * sizeof(ffm_float))
        ffm_importance_weights* iw = <ffm_importance_weights *> PyMem_Malloc(sizeof(ffm_importance_weights))

    iw.sum = sum(weights)
    iw.l = len(weights)
    for i, w in enumerate(weights):
        W[i] = w
    iw.W = W
    return iw


cdef void free_ffm_iw(ffm_importance_weights* iw):
    if iw is NULL:
        return
    if iw.W is not NULL:
        PyMem_Free(iw.W)
    PyMem_Free(iw)


cdef class _weights_finalizer:
    cdef void *_data

    def __dealloc__(self):
        if self._data is not NULL:
            free(self._data)


cdef object _train(
    ffm_problem* tr_ptr,
    ffm_problem* va_ptr,
    ffm_importance_weights* iw_ptr,
    ffm_importance_weights* iwv_ptr,
    ffm_parameter param,
):
    model_ptr = ffm_train_with_validation(tr_ptr, va_ptr, iw_ptr, iwv_ptr, param)
    if model_ptr is NULL or model_ptr.W is NULL:
        raise MemoryError("Invalid model pointer")

    best_iteration = model_ptr.best_iteration

    cdef:
        cnp.npy_intp shape[3]
        cnp.ndarray arr
        _weights_finalizer f = _weights_finalizer()

    shape = (model_ptr.n, model_ptr.m, model_ptr.k)
    arr = cnp.PyArray_SimpleNewFromData(3, shape, cnp.NPY_FLOAT32, model_ptr.W)

    # Note that `model_ptr.W` will be deallocated by Numpy array finalizer.
    f._data = <void*> model_ptr.W
    cnp.set_array_base(arr, f)
    free(model_ptr)
    return arr, best_iteration


def train(
    tr,
    va=None,
    iw=None,
    iwv=None,
    eta=0.2,
    lambda_=0.00002,
    nr_iters=15,
    k=4,
    nr_threads=1,
    auto_stop_threshold=-1,
    quiet=True,
    normalization=True,
    random=True,
    auto_stop=True,
):
    cdef ffm_parameter param
    param.eta = eta
    param.lambda_ = lambda_
    param.nr_iters = nr_iters
    param.k = k
    param.nr_threads = nr_threads
    param.auto_stop_threshold = auto_stop_threshold
    param.json_meta_path = NULL
    param.quiet = quiet
    param.normalization = normalization
    param.random = random
    param.auto_stop = auto_stop

    cdef:
        ffm_problem* tr_ptr = make_ffm_prob(tr[0], tr[1])
        ffm_problem* va_ptr
        ffm_importance_weights *iw_ptr, *iwv_ptr

    if va is not None:
        va_ptr = make_ffm_prob(va[0], va[1])
    else:
        va_ptr = NULL

    if iw is not None:
        iw_ptr = make_ffm_iw(iw)
    else:
        iw_ptr = NULL

    if iwv is not None:
        iwv_ptr = make_ffm_iw(iwv)
    else:
        iwv_ptr = NULL

    try:
        weights, best_iteration = _train(tr_ptr, va_ptr, iw_ptr, iwv_ptr, param)
    finally:
        free_ffm_prob(tr_ptr)
        free_ffm_prob(va_ptr)
        free_ffm_iw(iw_ptr)
        free_ffm_iw(iwv_ptr)
    return weights, best_iteration
