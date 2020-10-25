
from __future__ import division

import numpy as np
from numpy.random import random as _random, randint as _randint


def rosenbrock_fn(x):
    _x = np.array(x)
    return sum(100.0 * (_x[1:] - _x[:-1] ** 2) ** 2. + (1 - _x[:-1]) ** 2.)

class DiffEvolOptimizer(object):

    def __init__(self, fun, bounds, npop, F=0.5, C=0.9, seed=None, maximize=False):
       
        if seed is not None:
            np.random.seed(seed)

        self.fun = fun
        self.bounds = np.asarray(bounds)
        self.npop = npop
        self.F = F
        self.C = C

        self.ndim  = (self.bounds).shape[0]
        self.m  = -1 if maximize else 1

        bl = self.bounds[:, 0]
        bw = self.bounds[:, 1] - self.bounds[:, 0]
        self.population = bl[None, :] + _random((self.npop, self.ndim)) * bw[None, :]
        self.fitness = np.empty(npop, dtype=float)
        self._minidx = None

    def step(self):
        #Fazendo passos da otimização
        rnd_cross = _random((self.npop, self.ndim))
        for i in range(self.npop):
            t0, t1, t2 = i, i, i
            while t0 == i:
                t0 = _randint(self.npop)
            while t1 == i or t1 == t0:
                t1 = _randint(self.npop)
            while t2 == i or t2 == t0 or t2 == t1:
                t2 = _randint(self.npop)

            v = self.population[t0,:] + self.F * (self.population[t1,:] - self.population[t2,:])

            crossover = rnd_cross[i] <= self.C
            u = np.where(crossover, v, self.population[i,:])

            ri = _randint(self.ndim)
            u[ri] = v[ri]

            ufit = self.m * self.fun(u)

            if ufit < self.fitness[i]:
                self.population[i,:] = u
                self.fitness[i] = ufit

    @property
    def value(self):
        #Retorna o melhor valor da função otimizada
        return self.fitness[self._minidx]

    @property
    def location(self):
        #Localizacao do best fit
        return self.population[self._minidx, :]

    @property
    def index(self):
        return self._minidx

    def optimize(self, ngen=1):
    
        for i in range(self.npop):
            self.fitness[i] = self.m * self.fun(self.population[i,:])

        for j in range(ngen):
            self.step()

        self._minidx = np.argmin(self.fitness)
        return self.population[self._minidx,:], self.fitness[self._minidx]

    def iteroptimize(self, ngen=1):

        for i in range(self.npop):
            self.fitness[i] = self.m * self.fun(self.population[i,:])

        for j in range(ngen):
            self.step()
            self._minidx = np.argmin(self.fitness)
            yield self.population[self._minidx,:], self.fitness[self._minidx]

    def __call__(self, ngen=1):
        return self.iteroptimize(ngen)
    
ngen, npop, ndim = 100, 100, 2
limits = [[-1, 2]] * ndim

pop = np.zeros([ngen, npop, ndim])
loc = np.zeros([ngen, ndim]) 
de = DiffEvolOptimizer(rosenbrock_fn, limits, npop)
de(ngen)

for i, res in enumerate(de(ngen)):
    pop[i,:,:] = de.population.copy()
    loc[i,:] = de.location.copy()
    
print("Mínimos")
print(de.location)


