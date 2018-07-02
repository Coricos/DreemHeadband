# Author : Fujitsu
# Date : 01/07/2018

from cmaes.imports import *

# Returns the index of the least significant set bit 
# x is an integer 

def ffs(x):

    return int(x & -x).bit_length() - 1

# Returns a weighted sum scalarization (function)
# f is a map from iterable to iterable (awaitable)
# w is a weight
# z is an utopian point

def weighted_sum(f, w, z=None, **kwargs):

    if z is None: z = np.zeros(len(w))

    async def fun(x):

        fx = await f(x)
        return np.dot(w, fx - z)

    return fun

# Returns an epsilon constraint scalarization
# f is a map from iterable to iterable (awaitable)
# w[0] is an objective weight while w[1:] are constraint boundaries
# z is an utopian point

def epsilon_constraint(f, w, z=None, **kwargs):

    if z is None: z = np.zeros(len(w))

    async def fun(x):

        fx = await f(x)
        objective = w[0] * (fx[0] - z[0])
        penalty = np.sum([max(g, 0) for g in w[1:] - (fx[1:] - z[1:])])
        return objective + penalty

    return fun

# Defines a Tchebycheff norm scalarization
# f is a vector-valued function
# w is a weight
# z is an utopian point

def tchebycheff(f, w, z=None, **kwargs):

    if z is None: z = np.zeros(len(w))

    async def fun(x):

        fx = await f(x)
        return np.max(w * (fx - z))

    return fun

# Defines an augmented Tchebycheff norm scalarization
# f is a vector-valued function
# w is a weight
# z is an utopian point
# a are coefficients for weighted sum

def augmented_tchebycheff(f, w, z=None, a=1e-8, **kwargs):

    if z is None: z = np.zeros(len(w))

    async def fun(x):

        fx = await f(x)
        fx_z = fx - z
        return np.max(w * fx_z) + a * np.dot(w, fx_z)

    return fun

# Defines a penalty-based boundary intersection
# f is a vector-valued function
# w is a weight
# z is an utopian point
# a are coefficients for weighted sum

def pbi(f, w, z=None, a=0.25, **kwargs):

    if z is None: z = np.zeros(len(w))

    async def fun(x):

        fx = await f(x)
        fx_z = fx - z
        d1 = np.linalg.norm(np.dot(fx_z, w)) / np.linalg.norm(w)
        d2 = np.linalg.norm(fx_z - d1 * w)
        return d1 + a * d2

    return fun

# Defines an inverted penalty-based boundary intersection
# f is a vector-valued function
# w is a weight
# z is an utopian point
# a are coefficients for weighted sum

def ipbi(f, w, n=None, a=0.25, **kwargs):

    if n is None: n = np.ones(len(w))

    async def fun(x):

        fx = await f(x)
        n_fx = n - fx
        d1 = np.dot(n_fx, w) / np.linalg.norm(w)
        d2 = np.linalg.norm(n_fx - d1 * w)
        return -d1 + a * d2

    return fun

# Defines the L2-distance in a variable space
# a, b are points in the AWASpace

def dvar(a, b, **kwargs):

    return np.linalg.norm(a.x - b.x)

# Defines the L2-distance in the objective space
# a, b are points in the AWASpace

def dobj(a, b, **kwargs):

    return np.linalg.norm(a.y - b.y)

# Defines a zero function
# a, b are points in the AWASpace

def d0(a, b, **kwargs):

    return 0


# Defines the CMA-ES optimizer
# f is a scalar-valued objective function
# x0 is an initial solution to start optimization
# sigma0 is an initial standard deviation

async def cmaes(f, x0, sigma0=0.2, **kwargs):

    kwargs['verbose'] = -9
    kwargs['bounds'] = [0, 1]
    kwargs['seed'] = np.random.randint(2**32)

    es = cma.CMAEvolutionStrategy(x0, sigma0, inopts=kwargs)

    version1 = cma.__version__.split('.')[0] == '1'
    maxfevals = kwargs.get('maxfevals', float('+inf'))

    def es_result(i):

        return es.result()[i] if version1 else es.result[i]

    X = []

    while not es.stop():
        n = min(es.popsize, maxfevals - es.countevals)
        X = es.ask(n)
        tasks = []
        for x in X: tasks.append(await curio.spawn(f(x)))
        fX = await curio.gather(tasks)
        if n < es.popsize: break
        es.tell(X, fX)

    i = np.argmin([es.best.f] + fX)

    return es.best.x if i == 0 else X[i - 1]

# Convert string type to casted type
# s refers to a string object

def atot(s):

    if s == 'int': return int
    if s == 'float': return float

    raise ValueError("s must be 'int' or 'float'")
