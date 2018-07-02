# Author : Fujitsu
# Date : 01/07/2018

from cmaes.problem import *

# Representing iteration for m objective problems
# m is the number of objectives

def representing_iteration(m):

    return int(np.ceil(np.log2(m) + 1))

# Representing number for m objective problems
# m is the number of objectives

def representing_number(m):

    from moomin.awa import AddressSpace

    return len(AddressSpace(m, representing_iteration(m)))

# Problem optimization by AWA
# problem is an awa.Problem (objective to optimize)
# confopt is a dictionary containing the configurations
# logger is the logging.Logger

async def fmin(problem, confopt, logger=None):

    itr = confopt.get('max_iters', representing_iteration(problem.obj))
    evs = confopt.get('max_evals', {})
    evs = {Address(ast.literal_eval(k)):v for k, v in evs.items()}
    seed = confopt.get('seed')
    np.random.seed(seed)
    x0 = np.asarray(confopt.get('x0', np.random.rand(problem.obj, problem.dim) * 0.5 + 0.25))
    w0 = np.asarray(confopt.get('w0', np.eye(problem.obj)))
    scalarization = globals()[confopt.get('scalarization', 'weighted_sum')]
    optimization = globals()[confopt.get('optimization', 'cmaes')]
    awa = AWA(problem, x0, w0, s=scalarization, o=optimization, max_evals=evs)

    async def search_all():
        tasks = []
        for a in AddressSpace(problem.obj, itr): tasks.append(await awa.search_once(a))
        for t in tasks: await t.join()

    await search_all()

    return awa

# Initialize evaluators and returns a list of curio.Task
# params are the parameters for evaluators initialization
# queue is the asynchronous queue to pass data for evaluation
# logger is the logging.Logger

async def start_evaluators(params, queue, logger):

    lock = curio.Lock()
    tasks = []

    for worker in params:
        command = worker['command']
        tasks.append(await curio.spawn(evaluator(command, queue, lock, logger), daemon=True))

    return tasks

# Runs the infinite loop of evaluations
# command is the shell command to run
# queue refers to the curio.Queue
# lock refers to curio.Lock
# logger refers to logging.Logger

async def evaluator(command, queue, lock, logger):

    logger.info('%s, Initialized: %s', time.ctime(), command)

    while True:
        async with lock: prioity, io = await queue.get()
        cmd = io[0] + 'exec ' + command
        logger.info('%s, Start: %s', time.ctime(), cmd)
        stdout = await curio.subprocess.check_output(cmd, shell=True)
        logger.info('%s, End: %s', time.ctime(), cmd)
        io[1] = stdout.decode('utf-8')
        await queue.task_done()

# Defines an asynchronous version of the main function
# config refers to the given MooMin config file

async def amain(config):

    # Load configuration file
    with open(config) as f: conf = yaml.load(f)

    # Setup logger
    level = conf.get('verbose', logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    logger.addHandler(handler)

    # Setup parameters to be optimized
    params = conf['parameters']
    obj = conf.get('objectives', 1)
    use_param_names = conf.get('use_parameter_names', False)

    # Setup cache file
    cache_path = conf.get('cache', 'solutions.csv')
    queue = curio.Queue()
    problem = Problem(params, obj, cache_path, use_param_names, queue, logger)

    # Setup workers
    await start_evaluators(conf['workers'], queue, logger)

    # Run GA loop
    conf_opt = conf.get('optimizer', {})
    await fmin(problem, conf_opt, logger)
