from __future__ import division
from pyomo.environ import *
from pyomo.dae import *
import numpy as np
import pandas as pd

'''
Linear interpolation to find data for new time grid (e.g., the one obtained via FE collocation)
    Inputs: 
    - x_new: new domain for which to perform interpolation 
    - x: domain for which state data exists 
    - y: state measurement 
'''
def data_interp(x_new, x, y):
    from scipy import interpolate
    import numpy as np 
    
    f = interpolate.interp1d(x, y, kind='cubic')
    return f(x_new)


'''
Estimating data at the collocation points used to discretize the continuous time problem 
    Inputs:
    - t: first time period for the moving horizon iteration 
    - horizon_length: length of the time horizon of data availble and solution for each MH iteration
    - optim_options: options for discretization of continuous-time problem into discrete-time 
    - time_scale: time periods of the original state measurement 
    - y_measured: orignal state measurements collected 
'''
def time_scale_conversion(t, horizon_length, optim_options, time_scale, y_measured): 
     # Obtaining collocation time scale for current step
    m = ConcreteModel()
    m.t = ContinuousSet(bounds=(t, t + horizon_length))
    discretizer = TransformationFactory('dae.collocation')
    discretizer.apply_to(m, wrt=m.t, nfe=optim_options['nfe'], ncp=optim_options['ncp'])
    t_col = [t for t in m.t]

    # Interpolating data (cubic spline)
    y_col1 = data_interp(t_col, 
                         [i for i in time_scale if (i >= t) & (i <= t + horizon_length + 1e-2)],
                         [y_measured[i, 0] for i, j in enumerate(time_scale) if (j >= t) & (j <= t + horizon_length + 1e-2)])
    y_col2 = data_interp(t_col, 
                         [i for i in time_scale if (i >= t) & (i <= t + horizon_length + 1e-2)],
                         [y_measured[i, 1] for i, j in enumerate(time_scale) if (j >= t) & (j <= t + horizon_length + 1e-2)])
    
    y = np.column_stack((y_col1, y_col2))
    return y, t_col



'''
Formualates and solves NLP corresponding to one iteration of the MH discovery algorithm 
    Inputs: 
    - y_init: (vector 2x1) initial conditions for states at the beginning of the time horizon
    - t_span: (vecotr 2x1) time span for the optimization problem 
    - initial_theta: initial guess for the basis function coefficients (for first iteration these are obtained via OLS)
                     then the parameters from the previous iteration are used 
    - theta_bounds: bounds for the coefficients derived using the confidence intervals in OLS 
    - y: state data measurements 
    - iter_num: number of iteration in the moving horizon algorithm 
    - thresholded_indeces: indeces of the coefficients that already been set to zero 
    - optim_options: options for discretization of continuous-time problem into discrete-time 
'''
def optim_solve(y_init, t_span, initial_theta, theta_bounds, y, basis_0, basis_1, all_features_sym, iter_num, thresholded_indeces, optim_options, sign):
    
    print('\n')
    print('--------------------------- MH discovery: Iteration '+str(iter_num)+' ---------------------------\n')

    # Creating Pyomo model
    m = ConcreteModel()

    # Time horizon
    m.t = ContinuousSet(bounds=(t_span[0], t_span[1]))

    # Number of state variables
    m.n = RangeSet(1, 2)

    # Parameter for regularization
    m.alpha = Param(initialize=0.9)

    # Number of parameters to determine
    m.params = RangeSet(0, int(len(theta_bounds))-1)

    # Initializing the parameters
    m.theta_init = Param(m.params, initialize=initial_theta)

    # Absolute value regularization
    m.abs_reg_param = Param(initialize=50)

    # Variables (i.e., model parameters)
    def _bounds_rule1(m, i):
        return theta_bounds[i]

    def _init_rule(m, i):
        return m.theta_init[i]

    m.theta = Var(m.params, bounds=_bounds_rule1, initialize=_init_rule)

    # Absolute value variables
    def _bounds_rule2(m, i):
        bound_i = _bounds_rule1(m,i)
        return (0, max(abs(bound_i[0]), abs(bound_i[1])))

    m.abs_theta = Var(m.params, bounds=_bounds_rule2)

    # Defining state variables
    m.x = Var(m.n, m.t)
    
    # Adding rule to enfore positivity of the state and basis functions as e.g. 1/x can be considered, otherwise numerical issures arise
    if sign: 
        def _sign_rule(m,i,t): 
            return m.x[i,t] >= min(y[:,i-1])
        m.state_sign = Constraint(m.n, m.t, rule = _sign_rule)

    # Derivative variables
    m.dxdt = DerivativeVar(m.x, wrt=m.t)

    # Initial conditions
    for i in m.n:
        m.x[i, t_span[0]].fix(y_init[i - 1])

    # Differential equations in the model
    def _diffeq1(m, i, t):
        
        # Pyomo does not support numpy functions in it's constraints
        for n in basis_0['names']: 
            if n[0:3] == 'exp': 
                basis_0['functions'][basis_0['names'].index(n)] = lambda x,y: exp(x)
            elif n[0:3] == 'sin':
                basis_0['functions'][basis_0['names'].index(n)] = lambda x,y: sin(x)
            elif n[0:3] == 'cos':
                basis_0['functions'][basis_0['names'].index(n)] = lambda x,y: cos(x)
                
        for n in basis_1['names']: 
            if n[0:3] == 'exp': 
                basis_1['functions'][basis_1['names'].index(n)] = lambda x,y: exp(y)
            elif n[0:3] == 'sin':
                basis_1['functions'][basis_1['names'].index(n)] = lambda x,y: sin(y)
            elif n[0:3] == 'cos':
                basis_1['functions'][basis_1['names'].index(n)] = lambda x,y: cos(y)

            
        if i == 1:
            return m.dxdt[1, t] == sum(m.theta[i]*fun(m.x[1,t],m.x[2,t]) for i, fun in enumerate(basis_0['functions']))
        if i == 2:
            return m.dxdt[2, t] == sum(m.theta[i+len(basis_0['functions'])]*fun(m.x[1,t],m.x[2,t]) for i, fun in enumerate(basis_1['functions']))

    m.diffeq1 = Constraint(m.n, m.t, rule=_diffeq1)

    
    # Fixing thresholded parameters to zero 
    for i in thresholded_indeces:
        m.theta[i].fix(0)


    # Constraints for linearizing the absolute value of parameters theta
    def _theta_abs1(m, i):
        return m.theta[i] <= m.abs_theta[i]

    m.theta_abs1 = Constraint(m.params, rule=_theta_abs1)

    def _theta_abs2(m, i):
        return -m.theta[i] <= m.abs_theta[i]

    m.theta_abs2 = Constraint(m.params, rule=_theta_abs2)

    # Sum of absolute values of theta
    m.theta_sum_abs = Var(within=NonNegativeReals)
    m.theta_sum_abs_c = Constraint(expr=sum(m.abs_theta[i] for i in m.params) <= m.theta_sum_abs)

    # Sum of square values of theta
    m.theta_sum_sq = Var(within=NonNegativeReals)
    m.theta_sum_sq_c = Constraint(expr=sum(m.theta[i] ** 2 for i in m.params) <= m.theta_sum_sq)

    # Rate of change penalty
    m.theta_sum_diff_sq = Var(within=NonNegativeReals)
    m.theta_sum_diff_sq_c = Constraint(
        expr=sum((m.theta[i] - m.theta_init[i]) ** 2 for i in m.params) <= m.theta_sum_diff_sq)

    try:
        sim = Simulator(m, package='scipy')
        sim.simulate(integrator = 'dopri5', numpoints = 500 )

        # Discretize model using Collocation
        discretizer = TransformationFactory('dae.collocation')
        discretizer.apply_to(m, wrt=m.t, nfe=optim_options['nfe'], ncp=optim_options['ncp'])

        sim.initialize_model()
    except:
        print('\n')
        print('Simulation status: Could not use simulation to initialize the model')

        # Discretize model using Collocation
        discretizer = TransformationFactory('dae.collocation')
        discretizer.apply_to(m, wrt=m.t, nfe=optim_options['nfe'], ncp=optim_options['ncp'])


    xs = np.array([i for i in m.t])
    y_init_dict = {}
    for i in m.n:
        for k, j in enumerate(y):
            y_init_dict[i, xs[k]] = j[i - 1]

    m.y_data = Param(m.n, m.t, initialize=y_init_dict)

    # Squared error
    m.error_sq = Var(within=NonNegativeReals)
    m.error_sq_c = Constraint(
        expr=sum(sum((m.x[i, t_i] - m.y_data[i, t_i]) ** 2 for i in m.n) for t_i in m.t) <= m.error_sq)

    # Sum of objectives
    def sum_objs(m):
        return ((1 / len(y)) * m.error_sq + 0 * (m.alpha * m.theta_sum_sq + (1 - m.alpha) * m.theta_sum_abs) + 0 * m.theta_sum_diff_sq)

    m.obj = Objective(rule=sum_objs, sense=minimize)

    # CONOPT
    solver = SolverFactory('gams')
    results = solver.solve(m, tee=False)
    
    if results['Solver'][0]['Termination condition'] == 'infeasible': 
        theta_out = [initial_theta[i] for i in m.params]
    else: 
        theta_out = [value(m.theta[i]) for i in m.params]
    
    print('Solver status: '+results['Solver'][0]['Termination condition'],'\n')
    
    
    print('Coefficients:')
    for i in m.params:
        if value(m.theta[i]) == 0: 
            print('Theta %2s (%8s): 0' % (str(i), all_features_sym[i]))
        else: 
            print('Theta %2s (%8s): %.6e -- (%.6e, %.6e)' % (str(i), all_features_sym[i],value(m.theta[i]), theta_bounds[i][0], theta_bounds[i][1] ))

    return theta_out, value(m.error_sq)




# Simulated dynamics
def dyn_sim(theta, xs, y, basis_0, basis_1):
    from scipy.integrate import odeint
    
    def dy_dt_sim(y, t):

        dy_dt_sim = np.zeros(y.shape)
                
        for i in range(0, len(y)):
            
            for n in basis_0['names']: 
                if n[0:3] == 'exp': 
                    basis_0['functions'][basis_0['names'].index(n)] = lambda x,y: np.exp(x)
                elif n[0:3] == 'sin':
                    basis_0['functions'][basis_0['names'].index(n)] = lambda x,y: np.sin(x)
                elif n[0:3] == 'cos':
                    basis_0['functions'][basis_0['names'].index(n)] = lambda x,y: np.cos(x)
                
            for n in basis_1['names']: 
                if n[0:3] == 'exp': 
                    basis_1['functions'][basis_1['names'].index(n)] = lambda x,y: np.exp(y)
                elif n[0:3] == 'sin':
                    basis_1['functions'][basis_1['names'].index(n)] = lambda x,y: np.sin(y)
                elif n[0:3] == 'cos':
                    basis_1['functions'][basis_1['names'].index(n)] = lambda x,y: np.cos(y)


            if i == 0:
                dy_dt_sim[0] = sum(theta[i]*fun(y[0],y[1]) for i, fun in enumerate(basis_0['functions']))
            if i == 1:
                dy_dt_sim[1] = sum(theta[i+len(basis_0['functions'])]*fun(y[0],y[1]) for i, fun in enumerate(basis_1['functions']))
                
        return dy_dt_sim

    ys = odeint(dy_dt_sim, [y[0,0], y[0,1]], xs)
       
    return ys



'''
Function that performs thresholding step every couple of iterations 
    Inputs:
    - coeff_num: total number of coefficients (28 for the current 2D case)
    - thresholded_indices: indeces of coefficients that have already been set to zero 
    - theta_updates: history of the coefficients at all iterations performed thus far 
    - iter_num: current iteration number 
    - t_span:
    - y: 
    - iter_thresh: frequency of thresholding (i.e., every 20 iterations set to default)
    - tolerance: tolerance for signal to noise ratio for which parameters are eliminated 
'''
def thresholding_mean_to_std(coeff_num, thresholded_indices, theta_updates, iter_num, t, y, iter_thresh = 20, tolerance = 1): 
    # Determining parameters to threshold
    theta_avg = [] 
    if not iter_num % iter_thresh and iter_num > 0:
        for i in range(0,coeff_num):
            if i in [0,1,14,15,26,27] and iter_num/iter_thresh < 2: 
                pass 
            else: 
                if i not in thresholded_indices:
                    theta_current = []
                    for j in range(1 + int(iter_thresh*(iter_num/iter_thresh - 1)),iter_num):
                        theta_current.append(theta_updates[j][i])

                    # If ratio of mean to standard deviation does not meet the tolerance set parameter to zero
                    if abs(np.mean(theta_current)/np.std(theta_current)) < tolerance:
                        thresholded_indices.append(i)
                    theta_avg.append(np.mean(theta_current))
                else: 
                    theta_avg.append(0)
                    
        print('\n')
        print('Thresholded indices '+'('+str(len(thresholded_indices))+') :',thresholded_indices, '\n')
    
    return thresholded_indices


'''
Function that performs thresholding step every couple of iterations 
    Inputs:
    - coeff_num: total number of coefficients (28 for the current 2D case)
    - thresholded_indices: indeces of coefficients that have already been set to zero 
    - theta_updates: history of the coefficients at all iterations performed thus far 
    - iter_num: current iteration number 
    - t_span:
    - y: 
    - iter_thresh: frequency of thresholding (i.e., every 20 iterations set to default)
    - tolerance: tolerance for signal to noise ratio for which parameters are eliminated 
'''
def thresholding_accuracy_score(coeff_num, thresholded_indices, theta_updates, iter_num, t, y, iter_thresh = 20, tolerance = 1): 
    from sklearn.metrics import r2_score
    
    # Determining parameters to threshold
    theta_avg = [] 
    if not iter_num % iter_thresh and iter_num > 0:
        for i in range(0,coeff_num):
            if i not in thresholded_indices:
                theta_current = []
                for j in range(1 + int(iter_thresh*(iter_num/iter_thresh - 1)),iter_num):
                    theta_current.append(theta_updates[j][i])
                    
                theta_avg.append(np.mean(theta_current))
            else: 
                theta_avg.append(0)
                    
   
        
        y_sim = dyn_sim(theta_avg, t, y[:,0], y, False)
        r2_full = r2_score(y, y_sim)  
        flag = True 
        while flag: 
            theta_avg_red = [] 
            print(theta_avg)
            for k,i in enumerate(theta_avg): 
                if abs(i) != np.min([abs(theta_avg[j]) for j in np.nonzero(theta_avg)[0]]): 
                    theta_avg_red.append(i)
                else: 
                    cunrrent_index = k 
                    theta_avg_red.append(0)
            print(theta_avg_red)
                    
            y_sim = dyn_sim(theta_avg_red, t, y[:,0], y, False)
            r2_current = r2_score(y, y_sim)    
            print(r2_full, r2_current)
            
            if abs((r2_full-r2_current))/abs(r2_full) < 0.05: 
                theta_avg = theta_avg_red
                r2_full = r2_current
                thresholded_indices.append(k)
            else: 
                flag = False 
            
                
        
        print('\n')
        print('R^2: ', r2_current)
        print('Thresholded indices: ',thresholded_indices, '\n')
    
    return thresholded_indices