import numpy as np
from ypstruct import structure

# Genetic Algorithm
# 1. Initialization
# 2. Select parents & Crossover
# 3. Mutate Offsprings
# 4. Merge main population and Offsprings
# 5. Evaluate, Sort & Select
# 6. Go to Step 2, if it is needed

def run(problem, params):
    """
    ----Extract the information received through problem and params for future use----
    """
    # Problem Information
    costfunc = problem.costfunc
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax

    # Parameters
    maxit = params.maxit
    npop = params.npop
    pc = params.pc # proportion of children to main population
    nc = np.round(pc*npop/2)*2 # to round nc to the nearest multiple of two for later use
    gamma = params.gamma

    """
    ----INITIALIZATION----
    """
    # Empty Individual Template
    empty_individual = structure()
    empty_individual.postion = None # The position
    empty_individual.cost = None # The value of cost

    #BestSolution Ever Found
    bestsol = empty_individual.deepcopy() # Creates a deep copy of the empty_individual structure and stores it in bestsol, so that any change in bestsol will NOT affect empty_individual
    bestsol.cost = np.inf # To initialize, we use the worst value of the cost function. IN THE SPHERE CASE, a minimization problem, it would be INFINITY

    # Initialize Population
    pop = empty_individual.repeat(npop) # pop becomes an array of npop amount of the empty_individual structure
    for i in range (0,npop):
        pop[i].position = np.random.uniform(varmin, varmax, nvar)# set a random point for position to begin. Creates nvar uniformly distributed random numbers from varmin to varmax
        pop[i].cost = costfunc(pop[i].position) # cost field equals the output of the cost function of the random number assigned above
        if pop[i].cost < bestsol.cost: # THIS IS BECAUSE SPHERE IS A MINIMIZATION PROBLEM, WE ARE TRYING TO GET THE SMALLER VALUE, THE COMPARISON MAY CHANGE
            bestsol.pop[i].deepcopy() # so that changes to bestsol do NOT affect pop[i]
        
    # Best Cost per Iteration
    bestcost = np.empty(maxit) #creates an empty array with the maxit number of spaces

    """
    time to enter the evolution loop of our GA :)
    """

    for it in range (0, maxit): # start the for loop for the number of iterations
        
        popc = []# popc = population of children
        for k in range(nc//2): # nc = number of children, //=integer devision
            
            """
            ----SELECT PARENTS----
            """
            q = np.random.permutation(npop) # creates random permutation in array that contains numbers from 0, 1, 2, to npop-1
            p1 = pop[q[0]]
            p2 = pop[q[1]] # p1 and p2 are parents that are chosen randomly using the array q

            c1, c2 = crossover(p1, p2, gamma) # crossover function accepts two structures parents 1 and 2 for crossover





    # Output
    out = structure() 
    out.pop = pop #stores the output of pop

    return out

def crossover(p1, p2, gamma=0.1): # must return two offsprings, gamma is the bound limit for crossovers
    """
    ----CROSSOVER----
    """
    c1 = p1.deepcopy()
    c2 = p1.deepcopy() # creates copies of a parent to initialize them for crossover

    # Uniform Crossover method ahead
    alpha = np.random.uniform(-gamma, 1+gamma, *c1.position.shape) # * uses all members of shape as separate arguments
    c1.position = alpha*p1.position+ (1-alpha)*p2.position # Crossover Method
    c2.position = alpha*p2.position+ (1-alpha)*p1.position # Crossover Method
    return c1, c2

def mutate(x, mu, sigma): # unmutated solution x, mutation rate mu, step size / standard deviation sigma (for gaussian random distribution)
