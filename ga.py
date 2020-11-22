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
    nc = int(np.round(pc*npop/2)*2) # to round nc to the nearest multiple of two for later use
    gamma = params.gamma
    mu = params.mu
    sigma = params.sigma

    """
    ----INITIALIZATION----
    """
    # Empty Individual Template
    empty_individual = structure()
    empty_individual.postion = None # The position
    empty_individual.cost = None # The value of cost

    #BestSolution Ever Found
    bestsol = empty_individual.deepcopy() # Creates a deep copy of the empty_individual structure and stores it in bestsol, so that any change in bestsol will NOT affect empty_individual
    bestsol.cost = np.inf # To initialize, we use the worst value of the cost function. in our case, it would be infinity

    # Initialize Population
    pop = empty_individual.repeat(npop) # pop becomes an array of npop amount of the empty_individual structure
    for i in range (0,npop):
        pop[i].position = np.random.uniform(varmin, varmax, nvar)# set a random point for position to begin. Creates nvar uniformly distributed random numbers from varmin to varmax
        pop[i].cost = costfunc(pop[i].position) # cost field equals the output of the cost function of the random number assigned above
        if pop[i].cost < bestsol.cost: # WE ARE TRYING TO GET THE SMALLER VALUE meaning it's closer to the real stock prices
            bestsol = pop[i].deepcopy() # so that changes to bestsol do NOT affect pop[i]
        
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

            # Perform Crossover
            c1, c2 = crossover(p1, p2, gamma) # crossover function accepts two structures parents 1 and 2 for crossover

            # Perform Mutation
            c1 = mutate(c1, mu, sigma)
            c2 = mutate(c2, mu, sigma)

            # Apply Bounds to the positions
            apply_bound(c1, varmin, varmax)
            apply_bound(c2, varmin, varmax)

            # Evaluate first offspring
            c1.cost = costfunc(c1.position)
            if c1.cost < bestsol.cost:
                bestsol = c1.deepcopy() # because this is a minimization problem, the better solution would be the smaller one, so the < operator is used.

            # Evaluate second offspring
            c2.cost = costfunc(c2.position)
            if c2.cost < bestsol.cost:
                bestsol = c2.deepcopy() # because this is a minimization problem, the better solution would be the smaller one, so the < operator is used.

            """
            ----MERGE MAIN POPULATION AND OFFSPRING
            """
            # Add offspring to popc (children population which will now contain nc amount of children)
            popc.append(c1)
            popc.append(c2)

        # Merge, Sort, and Select
        pop = pop + popc # MERGE

        """
        ----EVALUATE SORT AND SELECT----
        """
        pop = sorted(pop, key=lambda x: x.cost) # the key of the sorting operation is an anonymous function that returns x.cost for any x in the population, USING THE COST VALUE AS THE KEY VALUE
        pop = pop[0:npop] # selects the highest sorted individuals to bring it back down to npop individuals in pop

        # Store Best cost
        bestcost[it] = bestsol.cost

        # Show iteration information
        print("Iteration {}: Best Cost = {}".format(it, bestcost[it]))


    # Output
    out = structure() 
    out.pop = pop #stores the output of pop
    out.bestsol = bestsol
    out.bestcost = bestcost
    
    return out

def crossover(p1, p2, gamma=0.1): # must return two offsprings, gamma is the bound limit for crossovers
    """
    ----CROSSOVER----
    """
    c1 = p1.deepcopy()
    c2 = p1.deepcopy() # creates copies of a parent to initialize them for crossover

    # Uniform Crossover method
    alpha = np.random.uniform(-gamma, 1+gamma, *c1.position.shape) # * uses all members of shape as separate arguments
    c1.position = alpha*p1.position+ (1-alpha)*p2.position # Crossover Method
    c2.position = alpha*p2.position+ (1-alpha)*p1.position # Crossover Method
    return c1, c2

def mutate(x, mu, sigma): # unmutated solution x, mutation rate mu (how many % genes to modify), step size / standard deviation sigma (for gaussian random distribution)
    """
    ----MUTATE----
    """
    y = x.deepcopy() # copy of original solution so we can modify it to the mutated form
    flag = np.random.rand(*x.position.shape) <= mu #<--------- This expression is used to decide whether or not to mutate a gene. It picks a random number from 0 to 1 and if less or equal to mu then it selects it for mutation.
    ind = np.argwhere(flag) # returns the indices of an array where the expression is true
    y.position[ind] += sigma*np.random.randn(*ind.shape) # executes the mutation on the indices where flag was TRUE by adding a random number with the standard deviation sigma.
    return y

def apply_bound(x, varmin, varmax): # compares position of x to varmin and varmax just incase it's out of bounds
    x.position = np.maximum(x.position, varmin) # if everything is fine after mutation, then x.position will stay the same. If its below varmin, it'll just be replaced by varmin.
    x.position = np.minimum(x.position, varmax)