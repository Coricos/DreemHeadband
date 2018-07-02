# Author : Fujitsu
# Date : 01/07/2018

from cmaes.toolbox import *

# Defines the optimization functions
# AWA - Adaptive Weighted Aggregation

class AddressSpace(object):

    def __init__(self, obj, itr):

        '''Create an address space.
        Parameters
        ----------
        obj: int > 0
            The number of objectives
        itr: int > 0
            The number of iterations
        '''

        self.obj = obj
        self.itr = itr

    def __iter__(self):

        '''Generate addresses in this address space.
        Returns
        -------
        adr: generator
            A generator yielding an address, per call, in this address space.
        '''

        def iterate(c, r):

            if len(c) == self.obj - 1: yield Address(c + (r,))
            else:
                for i in range(r, -1, -1): yield from iterate(c + (i,), r - i)

        yield from iterate((), 2**(self.itr - 1))

    def __len__(self):

        '''The number of addresses in this address space.'''

        return scipy.misc.comb(2**(self.itr - 1) + self.obj - 1, self.obj - 1, exact=True)

    def new(self):

        '''Generate new addresses in this address space.'''

        class New(object):

            '''A subspace of the address space, which contains all new addresses.'''

            def __init__(self, parent):

                self.parent = parent

            def __contains__(self, a):
                
                '''Check if the address is new in this address space.'''

                return (isinstance(a, Address) and
                        len(a) == self.parent.obj and
                        sum(a) == 2**(self.parent.itr - 1))

            def __iter__(self):

                '''Generate new addresses in this address space.'''

                return (a for a in self.parent if a in self)

            def __len__(self):
                
                '''The number of new addresses in this address space.'''

                return len(self.parent) - len(AddressSpace(self.parent.obj, self.parent.itr - 1))

        return New(self)

    def __contains__(self, a):

        '''Check if the address is in this address space.'''

        return (isinstance(a, Address) and
                len(a) == self.obj and
                sum(a) <= 2**(self.itr - 1))

    def nearest(self, a):

        '''Find the nearest address within this address space.
        Parameters
        ----------
        a: iterable
            The query.
            ``a`` must have the length equal to the dimension of
            this address space and must iterates nonnegative numbers.
        '''

        r = 2**(self.itr - 1)
        a = np.asarray(a)
        b = a * r / np.sum(a)
        c = np.asarray(np.rint(b), dtype=int)
        d = np.abs(b - c)
        ix = np.argsort(d)[::-1]
        
        for i in ix:
            e = r - np.sum(c)
            if e == 0: break
            c[i] += np.sign(e)

        return Address(c)

    def average(self, addresses, weights=None):

        '''Take (weighted) average of addresses, and
        find the nearest address to it in this address space.
        Parameters
        ----------
        addresses: iterable
            The addresses to average.
            This must iterate ``Address`` objects, each of which has
            the length equal to the dimension of this address space.
        weights: iterable, optional
            The weights for addresses.
        Returns
        -------
        nearest: Address
            The nearest address to the average in this address space
            (in the sense of the L1 distance).
        '''

        a = np.asarray(addresses)
        s = np.sum(a, axis=1)
        w = np.fromiter((np.prod(np.delete(s, i)) for i in range(len(s))), dtype=int)
        a *= w[:, np.newaxis]
        a = np.average(a, axis=0, weights=weights)

        return self.nearest(a)

    def print(self, a):

        s = 2**(self.itr - 1) // sum(a)

        print(tuple(s*c for c in a))

class Address(tuple):

    def __new__(cls, c=[]):

        '''Create an address with coefficients ``c``.
        Parameters
        ----------
        c: iterable, optional
            Coefficients of the address to create.
            The iterable must returns nonnegative values of ``int``
            that satisfy ``sum(c) == 2**n`` for some ``n >= 0``.
        '''

        c = np.fromiter(c, dtype=int)
        bits = np.bitwise_or.reduce(c)
        i = ffs(bits)

        return super().__new__(cls, np.right_shift(c, i))

    def __init__(self, _):

        self.__guide_pairs = None

    def arg_odd(self):
        
        '''Indices to odd coefficients of this address.
        Returns
        -------
        indices: list
            A list of ``int`` containing all indices of odd coefficients.
            The indices are sorted in ascending order.
        '''

        return [i for i, c in enumerate(self) if c % 2 == 1]

    def arg_even(self):

        '''Indices to positive even coefficients of this address.
        Returns
        -------
        indices: list
            A list of ``int`` containing all indices of positive even coefficients.
            The indices are sorted in ascending order.
        '''

        return [i for i, c in enumerate(self) if c > 0 and c % 2 == 0]

    def arg_zero(self):

        '''Indices to zero coefficients of this address.
        Returns
        -------
        indices: list
            A list of ``int`` containing all indices of zero coefficients.
            The indices are sorted in ascending order.
        '''

        return [i for i, c in enumerate(self) if c == 0]

    def birth_time(self):

        '''The iteration when this address is created.'''

        return int(np.log2(np.sum(self))) + 1

    def deg(self):

        '''The degree of this address.'''

        return len(self) - len(self.arg_zero()) - 1

    def guide_pair(self):

        '''Guide point pairs for this address.'''

        if self.__guide_pairs: return self.__guide_pairs
        e = self.arg_even()
        o = self.arg_odd()
        self.__guide_pairs = []
        b0 = [0] * len(self)
        for i, j in zip(o, [1, -1] * (len(o) // 2)):
            b0[i] = j
        for i in range(self.deg()):
            bi = b0[:]
            if 0 < i and i < len(o) - 1:
                bi[o[i]], bi[o[i - 1]] = bi[o[i - 1]], bi[o[i]]
            elif len(o) - 1 <= i:
                bi[o[-1]] += 2
                bi[e[i - len(o) + 1]] -= 2
            self.__guide_pairs.append(
                (Address(a - b for a, b in zip(self, bi)),
                 Address(a + b for a, b in zip(self, bi))))

        return self.__guide_pairs

    def guide(self):
        
        '''Generator of guide points for this address.'''

        for al, ar in self.guide_pair():
            yield al
            yield ar

class AWAState(object):

    def __init__(self, a=None, w=None, x=None, y=None, z=None, n_itr=None, n_eval=None, task=None):

        self.a = a
        self.w = w
        self.x = x
        self.y = y
        self.z = z
        self.n_itr = n_itr
        self.n_eval = n_eval
        self.task = task

    def __repr__(self):

        return 'AWAState(\n a={}\n w={}\n x={}\n y={}\n z={}\n n_itr={}\n n_eval={}\n task={})'.format(
            self.a, self.w, self.x, self.y, self.z, self.n_itr, self.n_eval, self.task)

class AWA(object):

    def __init__(self, f, x0, w0=None, s=augmented_tchebycheff, o=cmaes, d=d0, e=1e-3, max_evals=None):
        
        '''Create an AWA object.
        Parameters
        ----------
        f: callable
            The vector-valued objective function to optimize.
        x0: iterable
            A matrix of initial solutions.
            The matrix shape determines the problem dimensionality.
            When the matrix is ``m`` x ``n``, the objective function ``f`` is
            treated as a map from an n-D space to an m-D space (i.e.,
            m objectives, n variables).
        w0: iterable, optional
            A matrix of initial weights.
            The matrix shape must be ``m`` x ``m`` where the value of ``m`` is
            determined by the shape of ``x0``.
            By default, ``np.eye(m)`` is used.
        s: callable, optional
            A scalarization function.
            This function must accept two arguments, a vector-valued objective
            function ``f`` and a weight ``w``, and returns a scalar-valued
            objective function.
            By default, ``augmented_tchebycheff`` function is used.
        o: callable, optional
            An optimization function.
            This function must accept two arguments, a scalar-valued objective
            function ``f`` and an initial solution ``x``, and returns an
            optimal solution to ``f``.
            By default, ``cmaes`` function is used.
        d: callable, optional
            A puseudo-distance function.
            This function must accept two arguments, an address ``a`` and
            another address ``b``, and returns an puseudo-distance between them.
            By default, ``d0`` function is used.
        e: float, optional
            A tolerance of optimization.
            This value is used to stop the search for an address.
            AWA stops the search when d(a, a_old) <= e holds.
            By default, ``1e-3`` is used.
        '''

        self.f = f
        self.s = s
        self.o = o
        self.d = d
        self.e = e
        self.max_evals = max_evals if max_evals else {}
        self.states = {}
        self.lock = curio.Lock()

        obj = len(x0)
        if w0 is None:
            w0 = np.eye(obj)

        for a, w, x in zip(AddressSpace(obj, 1), w0, x0):
            self.states[a] = AWAState(a, w, x)

    def __getitem__(self, key):

        '''Get the search result for an address.'''

        key = Address(key)

        return curio.run(self.get(key))

    async def get(self, a):

        '''Get the search result for an address, asynchronously.'''

        task = await self.search_once(a)
        await task.join()

        return self.states[a]

    async def search_once(self, a):

        '''Search for an address if it has not been searched.'''

        if a not in self.states: self.states[a] = AWAState(a)

        async with self.lock:
            if self.states[a].task is None:
                self.states[a].task = await curio.spawn(self.search(a))

        return self.states[a].task

    async def search(self, a):

        '''Search for an address whatever it has been searched.'''

        await curio.sleep(0)  # await for spawning this task
        ts = []
        for b in a.guide(): ts.append(await self.search_once(b))
        for t in ts: await t.join()

        # initialize
        if a.deg() > 0:
            self.states[a].w = np.mean(
                [self.states[b].w for b in a.guide()], axis=0)
            self.states[a].x = np.mean(
                [self.states[b].x for b in a.guide()], axis=0)

        # relocate
        sigma0 = 1.0 / 6.0
        if a.deg():
            mean_distance = np.mean(
                [dvar(self.states[a], self.states[b]) for b in a.guide()])
            sigma0 = mean_distance / 3.0
        ev = self.max_evals.get(a, float('+inf'))
        while True:
            f = self.s(self.f, self.states[a].w)
            x0 = self.states[a].x[:]
            self.states[a].x = await self.o(
                f, x0, sigma0=sigma0, maxfevals=ev)
            if self.d(self.states[a].x, x0) <= self.e:
                self.states[a].y = await self.f(self.states[a].x)
                break
            self.weight_adaptation(a)

    def weight_adaptation(self, a):

        '''Adapt the weight of an address to make a new scalarized objective
        function has an optimal solution at an equi-distant place between each
        guide point pair.
        '''

        deg = a.deg()
        if deg == 0: return

        w = np.zeros(len(a))
        for al, ar in a.guide_pair():
            dl = self.d(self.states[a], self.states[al])
            dr = self.d(self.states[a], self.states[ar])
            if dl > dr:
                r = (dl + dr) / (2 * dl)
                w += r * self.states[a].w + (1 - r) * self.states[al].w
            elif dr > dl:
                r = (dl + dr) / (2 * dr)
                w += r * self.states[a].w + (1 - r) * self.states[ar].w
            else:
                w += self.states[a].w

        self.states[a].w = w / deg
