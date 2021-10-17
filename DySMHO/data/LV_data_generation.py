from __future__ import division
from scipy.integrate import odeint
from scipy import interpolate
import matplotlib.pyplot as plt
from pyomo.environ import *
from pyomo.dae import *
import numpy as np


# Defining dynamcis
def dy_dt(y, t):
    dy_dt = np.array([y[0] - 0.01 * y[0] * y[1], -y[1] + 0.02 * y[0] * y[1]])

    return dy_dt



# Generating data for given:
#   - xs: (numpy array) time grid
#   - initial_conditions: (list) initial conditions for y1 and y2
#   - plot: (boolean) output plot of data generated
def data_gen(xs, initial_conditions, noise, plot):
    y0 = np.array([initial_conditions[0], initial_conditions[1]])
    ys = odeint(dy_dt, y0, xs)
    ys[1:, 0] = ys[1:, 0] + np.random.normal(noise[0], noise[1], len(ys[1:, 0]))
    ys[1:, 1] = ys[1:, 1] + np.random.normal(noise[2], noise[3], len(ys[1:, 1]))

    if plot == True:
        plt.plot(xs, ys[:, 0], color='#d73027')
        plt.plot(xs, ys[:, 1], color='#fc8d59')
        plt.show()

    return xs, ys


# Linear interpolation to find data for new time grid (e.g., the one obtained via FE collocation)
def data_interp(x_new, x, y):
    f = interpolate.interp1d(x, y, kind='cubic')
    # return np.interp(x_new,x,y)
    return f(x_new)

# Function that computes the derivative using simple finite differneces method: 
def finite_differences(y,t): 
    
    dy_dt1= (y[2:,0] - y[0:-2,0])/(t[2:] - t[:-2])
    dy_dt2= (y[2:,1] -y[0:-2,1])/(t[2:] - t[:-2])
    
    dydt = np.column_stack((dy_dt1, dy_dt2))
    
    return dydt, t[1:-1]


# Function to evaluate derivative from data using collocation discretization
def derivative_eval(y_init, t_span, n, y, nfe, ncp):

    # Creating Pyomo model
    m = ConcreteModel()

    # Time horizon
    m.t = ContinuousSet(bounds=(t_span[0], t_span[1]))

    # Number of state variables
    m.n = RangeSet(1, n)

    # Defining state variables
    m.x = Var(m.n, m.t)

    # Derivative variables
    m.dxdt = DerivativeVar(m.x, wrt=m.t, initialize= 0)

    # Initial conditions
    for i in m.n:
        m.x[i, t_span[0]].fix(y_init[i - 1])

    # Discretize model using Collocation
    discretizer = TransformationFactory('dae.collocation')
    discretizer.apply_to(m, wrt=m.t, nfe=nfe, ncp=ncp)

    xs = np.array([i for i in m.t])
    y_init_dict = {}
    for i in m.n:
        for k, j in enumerate(y):
            y_init_dict[i, xs[k]] = j[i - 1]

    m.y_data = Param(m.n, m.t, initialize=y_init_dict)

    def _data_fit(m, i, t):
        return m.x[i, t] == m.y_data[i, t]

    m.data_fit = Constraint(m.n, m.t, rule=_data_fit)

    m.obj = Objective(expr=sum(sum(m.x[i,t] for i in m.n) for t in m.t), sense=minimize)

    # CONOPT
    solver = SolverFactory('gams')
    results = solver.solve(m, tee=True)


    dx1dt = np.array([value(m.dxdt[1, t])  for t in m.t])
    dx2dt = np.array([value(m.dxdt[2, t])  for t in m.t])
    dxdt = np.column_stack((dx1dt, dx2dt))

    return dxdt

