# DINDIN Meryll
# May 17th, 2018
# Dreem Headband Sleep Phases Classification Challenge

# Defines a way to deal with 1-D signals

class Levels:

    # Initialization
    # vec refers to a 1D numpy array
    def __init__(self, vec):

        # Defines the filtration
        self.simplex_up = gudhi.SimplexTree()
        self.simplex_dw = gudhi.SimplexTree()
        # Fullfill the simplexes
        for i in np.arange(len(vec)): 
            self.simplex_up.insert([i], filtration=vec[i])
            self.simplex_dw.insert([i], filtration=-vec[i])
        for i in np.arange(len(vec)-1): 
            self.simplex_up.insert([i, i+1], filtration=vec[i])
            self.simplex_dw.insert([i, i+1], filtration=-vec[i])
        # Initialize the filtrations
        self.simplex_up.initialize_filtration()
        self.simplex_dw.initialize_filtration()

    # Get both persistences from the signal
    # graph refers whether to display a graph or not
    def get_persistence(self, graph=False):

        # Computes the persistences
        dig_up = self.simplex_up.persistence()
        dig_dw = self.simplex_dw.persistence()

        if graph:
            plt.figure(figsize=(18,8))
            fig = gd.GridSpec(2,2)
            plt.subplot(fig[0,0])
            gudhi.plot_persistence_diagram(dig_up)
            plt.subplot(fig[1,0])
            gudhi.plot_persistence_barcode(dig_up)
            plt.subplot(fig[0,1])
            gudhi.plot_persistence_diagram(dig_dw)
            plt.subplot(fig[1,1])
            gudhi.plot_persistence_barcode(dig_dw)
            plt.tight_layout()
            plt.show()

        # Filters infinite values
        dig_up = np.asarray([[ele[1][0], ele[1][1]] for ele in dig_up if ele[1][1] < np.inf])
        dig_dw = np.asarray([[ele[1][0], ele[1][1]] for ele in dig_dw if ele[1][1] < np.inf])

        return dig_up, dig_dw

    # Defines the Betti curves out of the barcode diagrams
    # mnu, mnd refer to the minimal value for discretization
    # mxu, mxd refers to the maximal value for discretization
    # num_points refers to the amount of points to get as output
    def betti_curves(self, mnu, mxu, mnd, mxd, num_points=100):

        # Aims at barcode discretization
        def functionize(val, descriptor):

            # Temporary function
            def dirichlet(x):
                return 1 if (x > descriptor[0]) and (x < descriptor[1]) else 0

            # Vectorized function
            fun = np.vectorize(dirichlet)

            return fun(val)

        val_up = np.linspace(mnu, mxu, num=num_points)
        val_dw = np.linspace(mnd, mxd, num=num_points)
        v,w = np.zeros(num_points), np.zeros(num_points)
        u,d = self.get_persistence()

        for ele in u: v += functionize(val_up, ele)
        for ele in d: w += functionize(val_dw, ele)

        # Memory efficiency
        del val_up, val_dw, u, d

        return v, w
