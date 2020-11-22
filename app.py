import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import ga

# Cost Function / Sphere test function
def prices(x):

    # Retrieve the output from the neural network as well as the real answer

    # Apply some formula using x to the NN's output 

    # Save how far the result is from the real answer and return that value. The best solution will be the one with the lowest value.
    

# Problem Definition
problem = structure()
problem.costfunc = prices # cost function refers to sphere above
problem.nvar = 5 # structure space (number of decision variables)
problem.varmin = -10 # structure space (lower bound of the variables)
problem.varmax = 10 # Structure space (upper bound of the variables)


# GA Parameters
params = structure()
params.maxit = 100 # maximum # iterations of algorithm (common among evolutionary algorithms and meta-heuristics)
params.npop = 20 # initial population size
params.pc = 1 # proportion of children to parents
params.gamma = 0.1 # used for the bounds of the crossovers in ga.py
params.mu = 0.2 # mutation rate (% of genes in each child that will get mutated)
params.sigma = 0.1 # standard deviation for gaussian distribution for random number for mutation

# Run GA (will accept two arguments, problem and params)
out = ga.run(problem, params) # runs the GA that's defined in ga.py that we imported at the start, and outputs its value in the variable out


# Results
