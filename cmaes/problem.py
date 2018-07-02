# Author : Fujitsu
# Date : 01/07/2018

from cmaes.awa import *

# Defines the class Problem

class Problem(object):

    def __init__(self, params, obj, cache_path, use_param_names, queue, logger):

        self.dim = len(params)
        self.names = [v.get('name', 'x{}'.format(i + 1))
                      for i, v in enumerate(params)]
        self.types = [atot(v.get('type', 'float')) for v in params]
        self.mins = np.array([v.get('min', '-inf') for v in params])
        self.maxs = np.array([v.get('max', '+inf') for v in params])
        self.obj = obj
        self.cache_path = cache_path
        self.use_param_names = use_param_names
        self.queue = queue
        self.logger = logger
        self.load()

    def load(self):

        '''Load the cache from a file.
        Examples
        --------
        >>> p = Problem(params=[{}], obj=2, cache_path="./solutions.csv")
        >>> p.load()
        '''

        self.clear_cache()
        with open(self.cache_path, 'a+') as cache:
            cache.seek(0)
            for row in csv.reader(cache):
                # The last columns are objectives
                f = np.array([float(i) for i in row[-self.obj:]])
                # The previous columns are parameters
                x = [float(i) for i in row[-self.dim-self.obj:-self.obj]]
                # Add them to a dictionary
                self.add_cache(x, f)

    def clear_cache(self):

        self.cache = {}
        self.solution_id = 0

    def add_cache(self, x, f):

        '''Add a solution and its value to the cache, and increment the solution id.'''

        self.cache[tuple(x)] = f
        self.solution_id += 1

    def cached(self, x):
        
        '''Check if a solution is cached.
        Returns
        -------
        cached: bool
            ``True`` if ``x`` is cached,
            ``False`` otherwise.
        '''

        return tuple(x) in self.cache

    def get_cache(self, x):

        '''Get cached objective values.
        Returns
        -------
        value: numpy.ndarray
            Objectives.
        '''

        return self.cache[tuple(x)]

    def save(self, start_ctime, end_ctime, exit_code, x, f):

        '''Save evaluation information.
        Parameters
        ----------
        start_ctime: str
            A ctime when the evaluation started.
        end_ctime: str
            A ctime when the evaluation ended.
        exit_code: str
            A charactor that represents the exit code of evaluation.
        x: seq
            Variables.
        f: numpy.ndarray
            Objectives.
        '''

        if self.cached(x): return

        self.add_cache(x, f)
        with open(self.cache_path, 'a', newline='') as cache:
            writer = csv.writer(cache)
            writer.writerow(
                [self.solution_id, start_ctime, end_ctime, exit_code] \
                + [repr(i) for i in x] \
                + [repr(i) for i in f])

    async def __call__(self, x):

        '''Objective function.
        Parameters
        ----------
        x: numpy.ndarray
            Variables to evaluate.
        Returns
        -------
        value: float
            The value of the objective function at ``x``,
            or ``inf`` when ABEND.
        '''

        # Denormalize the feasible region from [0, 1] to [min, max]
        x = self.transform(x)

        start_time = time.time()
        start_ctime = time.ctime(start_time)
        self.logger.info('%s, Start evaluation: %s', start_ctime, x)

        # Check cache
        if self.cached(x):
            ret = self.get_cache(x)
            end_time = time.time()
            end_ctime = time.ctime(end_time)
            self.logger.info('%s, Cache hit %s', end_ctime, x)
            self.logger.info(
                '%s, End evaluation: %s -> %s, Elapsed: %d seconds',
                end_ctime, x, ret, end_time - start_time)
            return ret

        # Set default objective values: +inf
        ret = np.array([float('inf')] * self.obj)
        exit_code = "N"

        try:
            # Check lower/upper bounds
            if np.any(x < self.mins) or np.any(self.maxs < x):
                end_time = time.time()
                end_ctime = time.ctime(end_time)
                self.logger.info('%s, Out of bounds %s', end_ctime, x)
                self.logger.info(
                    '%s, End evaluation: %s -> %s, Elapsed: %d sec',
                    end_ctime, x, ret, end_time - start_time)
                exit_code = "O"
                self.save(start_ctime, end_ctime, exit_code, x, ret)
                return ret

            # Evaluate a solution
            x_repr = [repr(self.types[i](x[i])) for i in range(self.dim)]
            cmd = 'parameters="' + ' '.join(x_repr) + '"; '
            if self.use_param_names:
                for k, v in zip(self.names, x_repr):
                    cmd += k + '=' + v + '; '
            data = [cmd, None]  # [in, out]
            await self.queue.put((1, data))
            self.logger.info('%s, Queued: %s', time.ctime(), cmd)
            await self.queue.join()
            objectives = data[1].rstrip().split("\n")[-1]
            ret = np.array([float(y) for y in objectives.split()])

        except Exception as e:
            self.logger.warning(
                '%s, Exception: %s with %s', time.ctime(), x, e)
            exit_code = "E"

        # Finalize
        end_time = time.time()
        end_ctime = time.ctime(end_time)
        self.logger.info('%s, End evaluation: %s -> %s, Elapsed: %d seconds',
                    end_ctime, x, ret, end_time - start_time)
        self.save(start_ctime, end_ctime, exit_code, x, ret)

        return ret

    def transform(self, z):
        
        '''Denormalize the parameter scale from the internal representation
        [0, 1] to the original range [min, max].
        Parameters
        ----------
        z: numpy.ndarray
            A parameter vector to be denormalized.
        Returns
        -------
        value: float
            The denormalized value of ``x``.
        '''

        x = z * (self.maxs - self.mins) + self.mins

        return np.array([self.types[i](x[i]) for i in range(self.dim)])


