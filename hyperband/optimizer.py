# Author: DINDIN Meryll
# Date: 27/04/2018
# Company: Fujitsu

from hyperband.parameters import *

# Multiprocessed evaluation
# params refers to the set of parameter to use
# func refers to a partially defined function (try_params)
def evaluate_params(params, func):
                    
    return func(params)

# Defines the optimizer strategy

class Hyperband:
    
    # Initialization
    # get_params_function is the generic function calling the parameters space
    # try_params_function refers to the sklearn fit of called parameters
    # n_jobs will limit the multiprocessed used of multi-search
    def __init__(self, get_params_function, try_params_function, n_jobs=multiprocessing.cpu_count(), max_iter=100):

        self.get_params = get_params_function
        self.try_params = try_params_function
        self.n_jobs = n_jobs
        
        # Defines configuration downsampling rate (default=3)²
        self.eta = 3
        self.max_iter = max_iter           

        self.logeta = lambda x: np.log(x) / np.log(self.eta)
        self.s_max = int(self.logeta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter

        self.results = []
        self.counter = 0
        self.best_loss = np.inf
        self.best_counter = -1
        
    # Can be called multiple times
    # key refers to the name of the regressor used
    # data is a dictionnary whose keys points towards the data used by the models
    # skip_last for quicker bandit selection
    def run(self, key, data, skip_last=0):
        
        for s in reversed(range(self.s_max + 1)):
            
            # Initial number of configurations
            n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))   
            # Initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)
            # n random configurations
            T = [self.get_params(key) for i in range(n)] 
            
            for i in range((s + 1) - int(skip_last)):
                
                # Run each of the n configs for <iterations> 
                # Keep best (n_configs / eta) configurations
                n_configs = n * self.eta ** (-i)

                # Defines the partial function relative to evaluation
                fun = partial(self.try_params, key=key, data=data)

                # Multiprocessed bandit branches if needed
                pol = multiprocessing.Pool(processes=self.n_jobs)
                res = pol.map(partial(evaluate_params, func=fun), T)
                pol.close()
                pol.join()

                # Extract the val losses
                val_losses = np.asarray([ele['kappa'] for ele in res])
                    
                # Keeping track of the best result so far (for display only)
                if min(val_losses) < self.best_loss:
                    self.best_loss = min(val_losses)
                    self.best_counter = self.counter + val_losses.argmin()
                
                for idx in range(len(res)): res[idx]['params'] = T[idx]
                
                self.results += res
                
                # Select a number of best configurations for the next loop
                indices = np.argsort(val_losses)
                T = [T[i] for i in indices]
                T = T[0:int(n_configs / self.eta)]

                self.counter += len(T)
                print('\n{} | Best score so far: {} (run {})\n'.format(
                    self.counter, -self.best_loss, self.best_counter))
        
        return self.results
    