import numpy as np
import matplotlib.pyplot as plt
import scipy.odr
import random
import itertools
from itertools import combinations

from Graphe import *


class Regression:
    """
    Fit a data set with a given model.

    Attributes
    ----------
    ... : type
        ...
    """
    def __init__(self, Model):
        self.input = Model()
        self.model_function = self.input
        self.model = scipy.odr.Model(self.model_function)
        self.init_conds = self.input.init_conds
        self.str_model = self.input.expression
        self.params = None
        self.params_incertitude = None
        self.dim = self.init_conds.shape[0]

    def set_data(self, x, y, dx, dy):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.y_mean = np.mean(self.y)
        
    def set_data_from_csv(self, file):
        self.x, self.dx, self.y ,self.dy = np.loadtxt(file,skiprows=1,delimiter=',',unpack=True)
        self.y_mean = np.mean(self.y)
        
    def solve(self):
        try:
            data = scipy.odr.RealData(self.x, self.y, sx=self.dx, sy=self.dy)
            odr = scipy.odr.ODR(data, self.model, beta0 = self.init_conds)
            out = odr.run()
            self.out = out
            self.params = out.beta
            self.y_pred = self.model_function(self.params,self.x)
            self.params_incertitude = out.sd_beta
            return self.params, self.params_incertitude
        except:
            print(f"(solve) Please enter data.")

    def correlation_coefficient(self):
        return 1-np.sum((self.y-self.y_pred)**2)/np.sum((self.y-self.y_mean)**2)

    def show_results(self, decimal_precision = 3):
        print("*******************************************************")
        print(self.str_model)
        for i in range(self.dim):
            print('\t'+f"a{i+1} = {round(self.params[i], decimal_precision)} ± {round(self.params_incertitude[i], decimal_precision)}")
        print(f"R² = {100*self.correlation_coefficient():.3f}%")
        print("*******************************************************")
    
    def lin_plot(self, compare = False, intersect=(0,0)):
        try:
            graphe = Lin_XY()
            graphe.set_xlabel('$x$') ; graphe.set_ylabel('$y$')
            graphe.grid(which='major', color='black', alpha=.1)
            graphe.grid(which='minor', color='black', alpha=.1, ls=':')
            graphe.ax.minorticks_on()
            graphe.set_axis_interection(intersect)
            x_min = min(self.x-self.dx)
            x_max = max(self.x+self.dx)
            self.x_array = x_array = np.linspace(x_min, x_max, 500)
            self.y_array = y_array = self.model_function(self.params, x_array)
            graphe.plot(x_array,y_array, linewidth=1, color='blue', label='ODR Regression')

            def elementary_matrix(dim, i):
                m = np.zeros((dim,dim))
                m[i,i] = 1
                return m

            def ps(v):
                dim = v.shape[0]
                m = np.zeros((dim,dim))
                for i in range(dim):
                    m = m+v[i]*elementary_matrix(dim,i)
                return m

            matrix = []
            for tup in list(map(np.array,list(itertools.combinations_with_replacement(np.linspace(-1,1,10), self.dim)))):
                matrix.append(ps(tup))
            predict_with_error = lambda m: self.model_function(self.params+self.params_incertitude.dot(m), x_array)
            test = np.array([predict_with_error(m) for m in matrix])
            self.mi = mi = np.min(test,axis=0)
            self.ma = ma = np.max(test,axis=0)

            graphe.plot(x_array, mi, lw=.7,alpha=.7,ls=':',color='blue')
            graphe.plot(x_array, ma, lw=.7,alpha=.7,ls=':',color='blue')
            
            graphe.ax.errorbar(x=self.x, y=self.y, xerr=self.dx, yerr=self.dy,linestyle='None', marker='x', linewidth=.7, color='red',markersize=2)
            graphe.show()
        except:
            print(f"(plot) Please run regression.")
            
    def to_csv(self):
        arr = np.array([self.x_array,self.y_array,self.mi,self.ma]).T
        np.savetxt('regression_output.csv', arr, delimiter=',',header='x,y,y_min,y_max',comments='')

class Linear():
    def __init__(self):
        self.expression = "Model: a1.x+b"
        self.init_conds = np.array( [1,0] )
    
    def __call__(self, params, x):
        a,b = params
        return a*x+b

y = np.array(list(range(14)))
x = np.arange(17.98,16.94-.08,-.08)
x_err = .005
y_err = .1

regression_q1 = Regression(Linear)
regression_q1.set_data_from_csv('data.csv')
regression_q1.solve()
regression_q1.show_results()
regression_q1.lin_plot()
regression_q1.to_csv()