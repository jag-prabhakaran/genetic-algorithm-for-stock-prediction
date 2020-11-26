import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import ga

# Cost Function
def prices(x):

    # Retrieve the output from the neural network as well as the real answer, take the inputs to the neural networks

    # Apply some formula using x to the NN's output 
    # IDEA: for each input factor to the neural network, we multiply it by one value in the x[] array. Then, we can add, average, any operation to combine them and add it to our NN result
    # 

    # Save how far the result is from the real answer and return that value. The best solution will be the one with the lowest value.
    

# Problem Definition
problem = structure()
problem.costfunc = prices # cost function refers to prices above
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
