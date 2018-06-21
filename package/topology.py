# DINDIN Meryll
# May 17th, 2018
# Dreem Headband Sleep Phases Classification Challenge

from package.imports import *

# Computes associated persistent objects

class Filtration: 

    # Initialisation
    # vec refers to the multidimensional input vector
    def __init__(self, vec):

        # Dimension for simplex construction
        self.vec = vec
        # Creates the complex
        self.alpha = gudhi.AlphaComplex(points=self.vec)
        self.alpha = self.alpha.create_simplex_tree(max_alpha_square=250.0)

    # Computes the corresponding vertexes
    # neighbors refers to the amount of corresponding neighbors in the graph
    def vertexes(self, neighbors):

        # Defines the neighboring graph
        kdt = KDTree(self.vec, leaf_size=30, metric='euclidean')
        tmp = np.square(kdt.query(self.vec, neighbors, return_distance=True)[0])

        return np.sqrt(np.sum(tmp, axis=1) / neighbors)

    # Sublevel filtration
    # neighbors refers to the amount of corresponding neighbors in the graph
    def apply_filtration(self, neighbors=5):

        # Defines the filtration
        fil = gudhi.SimplexTree()
        vtx = self.vertexes(neighbors)
        r_f = self.alpha.get_filtration()

        # Apply incremental filtration
        for ele in r_f:
            val = max(ele[1], max([vtx[i] for i in ele[0]]))
            fil.insert(ele[0], filtration=val)
        # Save as attribute
        fil.set_dimension(self.vec.shape[1])
        fil.initialize_filtration()
        self.alpha = fil

        # Memory efficiency
        del fil, vtx, r_f

    # Compute the graph persistence
    def compute_persistence(self):

        self.apply_filtration()
        self.alpha.persistence()

    # Defines the Betti curves out of the barcode diagrams
    # dimension refers to dimension focus and diagram extraction
    # m_n, m_x refer to the minimal value for discretization
    # num_points refers to the amount of points to get as output
    # graph refers whether to display a graph or not
    def betti_curves(self, dimension, m_n=None, m_x=None, num_points=100, graph=False):

        # Aims at barcode discretization
        def functionize(val, descriptor):

            # Temporary function
            def dirichlet(x):
                return 1 if (x > descriptor[0]) and (x < descriptor[1]) else 0

            # Vectorized function
            fun = np.vectorize(dirichlet)

            return fun(val)

        # Compute persistence
        res = np.zeros(num_points)
        dig = self.alpha.persistence_intervals_in_dimension(dimension)
        dig = np.asarray([[ele[0], ele[1]] for ele in dig if ele[1] < np.inf])

        if m_n and m_x: 
            val = np.linspace(m_n, m_x, num=num_points)
        else:
            m_n, m_x = np.min(dig), np.max(dig)
            val = np.linspace(m_n, m_x, num=num_points)

        for ele in dig: res += functionize(val, ele)

        # Memory efficiency
        del dig, val

        if graph:
            plt.figure(figsize=(18,2))
            plt.plot(res, label='Sublevel Filtration - Betti Curve')
            plt.legend(loc='best')
            plt.show()

        return res

    # Defines the persistent landscapes of the diagrams
    # dimension refers to dimension focus and diagram extraction
    # m_n, m_x refer to the minimal value for discretization
    # nb_landscapes refers to the amount of landscapes to build
    # num_points refers to the amount of points to get as output
    # graph refers whether to display a graph or not
    def landscapes(self, dimension, m_n=None, m_x=None, nb_landscapes=10, num_points=100, graph=False):

        # Automated construction of the landscapes
        # n_landscapes refers to the amount of landscapes to build
        # num_points refers to the amount of points to get as output
        # m_n, m_x refer to the extrema for discretization
        def build_landscapes(dig, nb_landscapes, num_points, m_n, m_x):

            # Prepares the discretization
            lcd = np.zeros((nb_landscapes, num_points))

            # Observe whether absolute or relative
            if m_n and m_x:
                stp = np.linspace(m_n, m_x, num=num_points)
            else:
                m_n, m_x = np.min(dig), np.max(dig)
                stp = np.linspace(m_n, m_x, num=num_points)

            # Use the triangular functions
            for idx, ele in enumerate(stp):
                val = []
                for pair in dig:
                    b, d = pair[0], pair[1]
                    if (d+b)/2.0 <= ele <= d: val.append(d - ele)
                    elif  b <= ele <= (d+b)/2.0: val.append(ele - b)
                val.sort(reverse=True)
                val = np.asarray(val)
                for j in range(nb_landscapes):
                    if (j < len(val)): lcd[j, idx] = val[j]

            return lcd
        
        # Computes the persistent landscapes for both diagrams
        dig = self.alpha.persistence_intervals_in_dimension(dimension)
        dig = np.asarray([[ele[0], ele[1]] for ele in dig if ele[1] < np.inf])
        ldc = build_landscapes(dig, nb_landscapes, num_points, m_n, m_x)

        # Display landscapes if necessary
        if graph:
            plt.figure(figsize=(18,2))
            plt.title('Persistent Landscapes')
            for ele in ldc: plt.plot(ele)
            plt.show()

        return ldc        

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
    # graph refers whether to display a graph or not
    def betti_curves(self, mnu=None, mxu=None, mnd=None, mxd=None, num_points=100, graph=False):

        # Aims at barcode discretization
        def functionize(val, descriptor):

            # Temporary function
            def dirichlet(x):
                return 1 if (x > descriptor[0]) and (x < descriptor[1]) else 0

            # Vectorized function
            fun = np.vectorize(dirichlet)

            return fun(val)

        # Compute persistence
        v,w = np.zeros(num_points), np.zeros(num_points)
        u,d = self.get_persistence(graph=graph)

        if mnu and mxu and mnd and mxd:
            val_up = np.linspace(mnu, mxu, num=num_points)
            val_dw = np.linspace(mnd, mxd, num=num_points)
        else:
            mnu, mxu = np.min(u), np.max(u)
            mnd, mxd = np.min(d), np.max(d)
            val_up = np.linspace(mnu, mxu, num=num_points)
            val_dw = np.linspace(mnd, mxd, num=num_points)

        for ele in u: v += functionize(val_up, ele)
        for ele in d: w += functionize(val_dw, ele)

        # Memory efficiency
        del val_up, val_dw, u, d

        if graph:
            plt.figure(figsize=(18,2))
            plt.subplot(1,2,1)
            plt.plot(v)
            plt.subplot(1,2,2)
            plt.plot(w)
            plt.show()

        return v, w

    # Defines the persistent landscapes of the diagrams
    # mnu, mnd refer to the minimal value for discretization
    # mxu, mxd refers to the maximal value for discretization
    # nb_landscapes refers to the amount of landscapes to build
    # num_points refers to the amount of points to get as output
    # graph refers whether to display a graph or not
    def landscapes(self, mnu=None, mxu=None, mnd=None, mxd=None, nb_landscapes=10, num_points=100, graph=False):

        # Automated construction of the landscapes
        # n_landscapes refers to the amount of landscapes to build
        # num_points refers to the amount of points to get as output
        # m_n, m_x refer to the extrema for discretization
        def build_landscapes(dig, nb_landscapes, num_points, m_n, m_x):

            # Prepares the discretization
            lcd = np.zeros((nb_landscapes, num_points))

            # Observe whether absolute or relative
            if m_n and m_x:
                stp = np.linspace(m_n, m_x, num=num_points)
            else:
                m_n, m_x = np.min(dig), np.max(dig)
                stp = np.linspace(m_n, m_x, num=num_points)

            # Use the triangular functions
            for idx, ele in enumerate(stp):
                val = []
                for pair in dig:
                    b, d = pair[0], pair[1]
                    if (d+b)/2.0 <= ele <= d: val.append(d - ele)
                    elif  b <= ele <= (d+b)/2.0: val.append(ele - b)
                val.sort(reverse=True)
                val = np.asarray(val)
                for j in range(nb_landscapes):
                    if (j < len(val)): lcd[j, idx] = val[j]

            return lcd
        
        # Computes the persistent landscapes for both diagrams
        u,d = self.get_persistence(graph=graph)
        l_u = build_landscapes(u, nb_landscapes, num_points, mnu, mxu)
        l_d = build_landscapes(d, nb_landscapes, num_points, mnd, mxd)

        # Display landscapes if necessary
        if graph:
            plt.figure(figsize=(18,2))
            plt.subplot(1,2,1)
            for ele in l_u: plt.plot(ele)
            plt.subplot(1,2,2)
            for ele in l_d: plt.plot(ele)
            plt.show()

        return l_u, l_d

        