from __future__ import division
from pyomo.environ import *
from pyomo.dae import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from importlib import reload
from sklearn import linear_model


class DySMHO(): 
    
    '''
    Initiating class with data and basis functions:
        Inputs: 
        - y: N_t by 2 array of noisy state measurements (N_t is the number of measurements)
        - t: is the times at which the measurements in y were collected (arbitrary units)
        - basis: list of length 2 with disctionaries corresponding to basis functions used for identification of the dynamics
    ''' 
    def __init__(self, y, t, basis):
        self.y = y
        self.t = t 
        self.basis_0 = basis[0]
        self.basis_1 = basis[1]
        
    '''
    Smoothing function applies the Savitzky-Golay filter to the state measurements
        Inputs:
        - window_size: (interger) The length of the filter window (i.e., the number of coefficients)
        - poly_order: (interger) The order of the polynomial used to fit the samples
        - verbose: (True/False) display information regarding smoothing iterations
    '''   
    def smooth(self, window_size = None , poly_order = 2, verbose = True): 
        from scipy.signal import savgol_filter
        from statsmodels.tsa.statespace.tools import diff
        
        if verbose: 
            print('\n')
            print('--------------------------- Smoothing data ---------------------------')
            print('\n')
            
        # Automatic tunning of the window size 
        if window_size == None: 
            
            y_norm0 = (self.y[:,0]-min(self.y[:,0]))/(max(self.y[:,0])-min(self.y[:,0]))
            self.smoothed_vec_0 = [y_norm0] 
            std_prev = np.std(diff(y_norm0,1))
            window_size_used = 1 
            std1 = [] 
            while True:
                std1.append(std_prev)
                window_size_used += 10
                y_norm0 = savgol_filter(y_norm0, window_size_used, poly_order)
                std_new = np.std(diff(y_norm0,1))
                if verbose: 
                    print('Prev STD: %.5f - New STD: %.5f - Percent change: %.5f' % (std_prev, std_new, 100*(std_new-std_prev)/std_prev))
                if abs((std_new-std_prev)/std_prev) < 0.1: 
                    window_size_used -= 10
                    break
                else:
                    std_prev = std_new
                    self.smoothed_vec_0.append(y_norm0)
                    y_norm0 = (self.y[:,0]-min(self.y[:,0]))/(max(self.y[:,0])-min(self.y[:,0]))  
                
            if window_size_used > 1: 
                print('Smoothing window size (dimension 1): '+str(window_size_used),'\n')
                
                self.y[:,0] = savgol_filter(self.y[:,0], window_size_used, poly_order)
            else: 
                print('No smoothing applied')
                print('\n')
                              
            
            y_norm1 = (self.y[:,1]-min(self.y[:,1]))/(max(self.y[:,1])-min(self.y[:,1])) 
            self.smoothed_vec_1 = [y_norm1] 
            std_prev = np.std(diff(y_norm1,1))
            window_size_used = 1 
            std2 = [] 
            while True:
                std2.append(std_prev)
                window_size_used += 10 
                y_norm1 = savgol_filter(y_norm1, window_size_used, poly_order)
                std_new = np.std(diff(y_norm1,1))
                if verbose: 
                    print('Prev STD: %.5f - New STD: %.5f - Percent change: %.5f' % (std_prev, std_new, 100*(std_new-std_prev)/std_prev))
                if abs((std_new-std_prev)/std_prev)  < 0.1: 
                    window_size_used -= 10
                    break   
                else:
                    std_prev = std_new
                    self.smoothed_vec_1.append(y_norm1)
                    y_norm1 = (self.y[:,1]-min(self.y[:,1]))/(max(self.y[:,1])-min(self.y[:,1])) 
                         
                 

            if window_size_used > 1: 
                print('Smoothing window size (dimension 2): '+str(window_size_used),'\n')
                self.y[:,1] = savgol_filter(self.y[:,1], window_size_used, poly_order)
            else: 
                print('No smoothing applied')
                print('\n')
#             print(std1)
#             print(std2)
        
        # Pre-specified window size
        else: 
            self.y[:,0] = savgol_filter(self.y[:,0], window_size, poly_order)
            self.y[:,1] = savgol_filter(self.y[:,1], window_size, poly_order)
            
            self.t = self.t[:len(self.y)]
            
    '''
    First pre-processing step which includes Granger causality analysis for derivative and basis functions 
        Inputs: 
        - granger: (boolean) whether Granger causality test is performed to filter the original basis or not 
        - significance: (real, lb = 0, ub = 1) significance level for p-values obatined ivia Granger causality test 
        - verbose: (boolean) display information regarding Granger tests 
        - rm_features: (list) feature names to remove after thresholding is performed
    '''
    def pre_processing_1(self, 
                         granger = True, 
                         significance = 0.1,
                         verbose = True, 
                         rm_features = [[],[]]): 
        
        # Computing derivatives using finite differences 
        dy_dt1= (self.y[2:,0] - self.y[0:-2,0])/(self.t[2:] - self.t[:-2])
        dy_dt2= (self.y[2:,1] - self.y[0:-2,1])/(self.t[2:] - self.t[:-2])
        dydt = np.column_stack((dy_dt1, dy_dt2))
        self.t_diff = self.t[:-1]
       
        
        # Generating features in pandas dataframe
        df_y1 = pd.DataFrame() 
        for i, basis_fun_i in enumerate(self.basis_0['functions']): 
            df_y1[self.basis_0['names'][i]] = [basis_fun_i(j[0],j[1]) for j in self.y[1:-1]]
            
        df_y1['dy_dt'] = (self.y[2:,0] - self.y[0:-2,0])/(self.t[2:] - self.t[:-2])
        df_y1.drop(df_y1.tail(1).index,inplace=True)
        df_y1['y_shift'] = self.y[2:-1,0]
        self.df_y1 = df_y1
        self.dy1_dt = (self.y[2:,0] - self.y[0:-2,0])/(self.t[2:] - self.t[:-2])
        self.y = self.y[1:-1]
        self.t = self.t[1:-1]
        
        
        df_y2 = pd.DataFrame() 
        for i, basis_fun_i in enumerate(self.basis_1['functions']): 
            df_y2[self.basis_1['names'][i]] = [basis_fun_i(j[0],j[1]) for j in self.y[1:-1]]
            
        df_y2['dy_dt'] = (self.y[2:,1] - self.y[0:-2,1])/(self.t[2:] - self.t[:-2])
        df_y2.drop(df_y2.tail(1).index,inplace=True)
        df_y2['y_shift'] = self.y[2:-1,1]
        self.df_y2 = df_y2
        self.dy2_dt = (self.y[2:,1] - self.y[0:-2,1])/(self.t[2:] - self.t[:-2])
        
        self.all_features_y1 = df_y1.columns
        self.all_features_y2 = df_y2.columns
        
        self.columns_to_keep1 = []
        self.columns_to_keep2 = []
        
        if '1' in self.df_y1.columns: 
            self.columns_to_keep1.append('1')
        if '1' in self.df_y2.columns: 
            self.columns_to_keep2.append('1')
            
        self.dy1_dt = df_y1['dy_dt']
        self.dy2_dt = df_y2['dy_dt']
        
        
        if granger: 
            from statsmodels.tsa.stattools import grangercausalitytests
            tests = ['ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest']

            gragner_causality = {}
            for i in df_y1.columns: 
                if i != '1': 
                    x = df_y1[i].dropna()
                    y = df_y1['y_shift'].dropna()
                    data = pd.DataFrame(data = [y,x]).transpose()
                    x = grangercausalitytests(data, 1, addconst=True, verbose=False)
                    p_vals = [x[1][0][test][1] for test in tests]

                    gragner_causality[i] = [np.mean(p_vals), np.std(p_vals)]

            df1 = pd.DataFrame.from_dict(gragner_causality).T 
            count = 0 
            for i in df1.index: 
                if df1[0][i] < significance and i != 'dy_dt': 
                    self.columns_to_keep1.append(i)
                count += 1

            gragner_causality = {}
            for i in df_y2.columns:
                if i != '1': 
                    x = df_y2[i]
                    y = df_y2['y_shift']
                    data = pd.DataFrame(data = [y,x]).transpose()
                    x = grangercausalitytests(data, 1, addconst=True, verbose=False)
                    p_vals = [x[1][0][test][1] for test in tests]

                    gragner_causality[i] = [np.mean(p_vals), np.std(p_vals)]

            df2 = pd.DataFrame.from_dict(gragner_causality).T
            count = 0 
            for i in df2.index: 
                if df2[0][i] < significance and i != 'dy_dt': 
                    self.columns_to_keep2.append(i)
                count += 1 

            if verbose: 
                print('\n')
                print('--------------------------- Pre-processing 1: Dimension 1 ---------------------------')
                print(df1,'\n')
                print('Columns to keep for y1: ', self.columns_to_keep1)
                
                print('\n')
                print('--------------------------- Pre-processing 2: Dimension 2 ---------------------------')
                print(df2,'\n')
                print('Columns to keep for y2: ', self.columns_to_keep2)
                
            df_y1.drop([i for i in df_y1.columns if i not in self.columns_to_keep1], axis = 1, inplace = True )
            df_y2.drop([i for i in df_y2.columns if i not in self.columns_to_keep2], axis = 1, inplace = True )
            
        
        self.df_y1 = df_y1
        self.df_y2 = df_y2
        
        for i in rm_features[0]: 
            if i in self.columns_to_keep1: 
                self.columns_to_keep1.remove(i)
        
        for i in rm_features[1]: 
            if i in self.columns_to_keep2: 
                self.columns_to_keep2.remove(i)
                
         
        
        
    '''
    Second pre-processing step which includes Ordinary Least Squares (OLS) for derivative and basis functions 
        Inputs: 
        - verbose: (boolean) print outputs of OLS 
        - plot:  (boolean) plot derivatives and resulting fit 
        - significance: (real, lb = 0, ub = 1) significance level for p-values obatined via OLS to determine non-zero coefficients 
        - confidence: (real, lb = 0, ub = 1) confidence level used to derive bounds for the non-zero parameters identified in OLS 
    '''
    def pre_processing_2(self, verbose = True, plot = False, significance = 0.9, confidence = 1-1e-8, initial_pre = True): 

        import statsmodels.api as sm
        from statsmodels.sandbox.regression.predstd import wls_prediction_std
        import cvxopt
        
        self.df_y1.drop([i for i in self.df_y1.columns if i not in self.columns_to_keep1], axis = 1, inplace = True )
        X_train = self.df_y1.to_numpy() 
        y_train = self.dy1_dt.to_numpy() 
        
        model = sm.OLS(y_train,X_train)
        results1 = model.fit()
        if verbose: 
            print('\n')
            print('--------------------------- Pre-processing 2: Dimension 0 ---------------------------\n')
            print(results1.summary()) 
            
        if plot: 
            prstd, iv_l, iv_u = wls_prediction_std(results1)
            plt.figure() 
            plt.plot(y_train, color = '#d73027', linewidth = 3)
            gray = [102/255, 102/255, 102/255]
            plt.plot(np.dot(X_train, results1.params), color = 'k', linewidth = 3)
            plt.legend(['Derivative data','Model prediction'])
            plt.title('OLS $y_0$')
            plt.show()
            
        self.df_y2.drop([i for i in self.df_y2.columns if i not in self.columns_to_keep2], axis = 1, inplace = True )
        X_train = self.df_y2.to_numpy() 
        y_train = self.dy2_dt.to_numpy() 

        model = sm.OLS(y_train,X_train)
        results2 = model.fit()
        if verbose: 
            print('\n')
            print('--------------------------- Pre-processing 2: Dimension 1 ---------------------------\n')
            print(results2.summary())
            print('\n','--------------------------- Pre-processing: FINISHED ---------------------------','\n \n')
            
        if plot: 
            prstd, iv_l, iv_u = wls_prediction_std(results2)
            plt.figure() 
            plt.plot(y_train, color = '#d73027', linewidth = 3)
            gray = [102/255, 102/255, 102/255]
            plt.plot(np.dot(X_train, results2.params), color = 'k', linewidth = 3)
            plt.legend(['Derivative data','Model prediction'])
            plt.title('OLS $y_1$')
            plt.show()
            
            
        initial_parameters = [] 
        bounds = []
        non_zero = [] 
        p_val_tolerance = significance
        confidence_interval = 1 - confidence
        all_features_sym = [] 

        # Start with count equal to 1 because first parameter is the constant term 
        conf_interval1 = results1.conf_int(alpha = confidence_interval) 
        count = 0
        count_vars = 0 
    
        
        for i in self.all_features_y1: 
            if i not in ['dy_dt','y_shift']: 
                all_features_sym.append(i)
                if (i in self.columns_to_keep1):
                    if (results1.pvalues[count]) < p_val_tolerance or (i in ['1','y0','y1'] and initial_pre == True): 
                        initial_parameters.append(results1.params[count])
                        bounds.append((conf_interval1[count][0],conf_interval1[count][1]))
                        non_zero.append(count_vars)
                    else: 
                        initial_parameters.append(0)
                        bounds.append((0,0))
                    count += 1


                elif (i not in self.columns_to_keep1): 
                    initial_parameters.append(0)
                    bounds.append((0,0))

                count_vars += 1 

    
        conf_interval2 = results2.conf_int(alpha = confidence_interval) 
        count = 0
        for i in self.all_features_y2: 
            if i not in ['dy_dt','y_shift']: 
                all_features_sym.append(i)
                if (i in self.columns_to_keep2):
                    if (results2.pvalues[count]) < p_val_tolerance or (i in ['1','y0','y1'] and initial_pre == True): 
                        initial_parameters.append(results2.params[count])
                        bounds.append((conf_interval2[count][0],conf_interval2[count][1]))
                        non_zero.append(count_vars)
                    else: 
                        initial_parameters.append(0)
                        bounds.append((0,0))
                    count += 1

                elif (i not in self.columns_to_keep2): 
                    initial_parameters.append(0)
                    bounds.append((0,0))

                count_vars += 1 
        
        self.initial_theta = initial_parameters
        self.theta_bounds = bounds 
        self.non_zero = non_zero 
        self.all_features_sym = all_features_sym
    
        
    '''
    Performs moving horizon dicovery routine
        Inputs: 
        - horizon_length: (int) number of sampling time steps to use for the optimization horizon 
        - time_steps: (int) number of time steps to perform moving horizon optimization
        - data_step: (int) length of the step (in number of data points) taken at each moving horizon iteration 
        - optim_options: (dict) options for the discretization
            - 'nfe': number of finite elements to use 
            - 'ncp': number of colocation points 
        - thresholding_tolerance: (float) coefficient of variation tolerance for thresholding basis functions 
    '''
    def discover(self, 
                 horizon_length, 
                 time_steps, data_step, 
                 optim_options = {'nfe':50, 'ncp':15}, 
                 thresholding_frequency = 20, 
                 thresholding_tolerance = 1,
                 sign = False): 
        
        y_init = self.y[0,:]
        y0_step = self.y[0:len(self.y) + 1:data_step]
        
        # Initializing iterations and error
        iter_num = 0
        thresholded_indices = [i for i,j in enumerate(self.initial_theta) if i not in self.non_zero ]
        len_thresholded_indices_prev = [len(thresholded_indices), 0]
        theta_init_dict = {i:j for i,j in enumerate(self.initial_theta)}
        error = []
        theta_updates = {0: self.initial_theta}
        # Number of thresholded terms after every iteration 
        self.number_of_terms = [len(thresholded_indices)]
        # Parameter values after each OLS step
        self.theta_after_OLS = [self.initial_theta]
        self.CV =[] 
        
        for k, t in enumerate(self.t[0:len(self.t) - 1:data_step]):

            if t + horizon_length < self.t[-1]:
                
                # Obtaining collocation time scale for current step
                from utils import time_scale_conversion
                y, t_col = time_scale_conversion(t, 
                                                 horizon_length, 
                                                 optim_options, 
                                                 self.t, 
                                                 self.y)

                
                # Performing optimization to compute the next theta
                from utils import optim_solve
                theta_init, error_sq = optim_solve(y_init, 
                                                   [t, t + horizon_length], 
                                                   theta_init_dict, 
                                                   self.theta_bounds, 
                                                   y, 
                                                   self.basis_0, 
                                                   self.basis_1, 
                                                   self.all_features_sym, 
                                                   iter_num, 
                                                   thresholded_indices, 
                                                   optim_options,
                                                   sign)
                error.append(error_sq)
        

                # Updating theta
                theta_updates[iter_num] = theta_init
                theta_init_dict = {i:j for i,j in enumerate(theta_init)} 
                
                # Determining parameters to threshold
                from utils import thresholding_accuracy_score, thresholding_mean_to_std
                thresholded_indices, CV = thresholding_mean_to_std(len(self.initial_theta), 
                                                               thresholded_indices, 
                                                               theta_updates, 
                                                               iter_num, 
                                                               self.t, 
                                                               self.y,
                                                               iter_thresh = thresholding_frequency, 
                                                               tolerance = thresholding_tolerance)
                self.number_of_terms.append(len(thresholded_indices))
                if len(CV)>0: 
                    self.CV.append(CV)
               
                # Beaking loop is the thresholded parametrs have not changed in 4 rounds of thresholding
                print('\n')
                if len(thresholded_indices) == len_thresholded_indices_prev[0]:
                    if len_thresholded_indices_prev[1] < 4*thresholding_frequency: 
                        len_thresholded_indices_prev[1] += 1 
                    else: 
                        break
                else: 
                    len_thresholded_indices_prev[0] = len(thresholded_indices) 
                    len_thresholded_indices_prev[1] = 0 
                    
                
                # Recomputing bounds once some of the parameters have been eliminated
                if not iter_num % thresholding_frequency and iter_num > 0:
                    
                    # Dropping columns in the dataframe containing the evaluated basis functions 
                    self.df_y1.drop([j for i,j in enumerate(self.all_features_sym) if (i < len(self.basis_0['functions'])) 
                                     and (i in thresholded_indices and j in self.df_y1.columns)], axis = 1, inplace = True )
                    self.df_y2.drop([j for i,j in enumerate(self.all_features_sym) if (i >= len(self.basis_0['functions'])) 
                                     and (i in thresholded_indices and j in self.df_y2.columns)], axis = 1, inplace = True )
                
                        
                    self.columns_to_keep1 = self.df_y1.columns
                    self.columns_to_keep2 = self.df_y2.columns
                    
                    # Running pre-processing again (OLS) -- to obatin better bounds for the parameters that remain 
                    self.pre_processing_2(verbose = True, 
                                          plot = False, 
                                          significance = 0.9, 
                                          confidence = 1-1e-8, 
                                          initial_pre = False)
                    thresholded_indices = [i for i,j in enumerate(self.initial_theta) if i not in self.non_zero ]
                    theta_init_dict = {i:j for i,j in enumerate(self.initial_theta)}
                    theta_updates[iter_num+1] = self.initial_theta
                    self.theta_after_OLS.append(self.initial_theta)
                    theta_init_dict = {i:j for i,j in enumerate(self.initial_theta)}
                
    

                # Obtaining the next initial condition
                if k + 1 < len(self.y):
                    y_init = [y0_step[k + 1, 0], y0_step[k + 1, 1]]

                iter_num += 1
                
        self.theta_values = theta_updates
        
    '''
    Performs validation of the discovered governing equations (integrating the resulting ODE/PDE and computing performance metrics and plotting) 
        Inputs: 
        - xs_validate: (list or array) time domain data for the validation set 
        - y_validate: (list or array) states (measured or simulated) used to validate predictions from the discovered model 
        - metric: (str) currently the only supported metric is 'MSE'
        - plot: (boolean) displaying validation plots 
    '''        
        
    def validate(self, xs_validate, y_validate, metric = 'MSE', plot = True): 
        
        theta_values = pd.DataFrame(self.theta_values)
        theta_values.loc[theta_values.iloc[:,-1] == 0, :] = 0
        mean_theta = theta_values.iloc[:,-30:-1].mean(axis=1).to_numpy()
        
        import utils 
        from utils import dyn_sim
        ys_mhl = dyn_sim(mean_theta, 
                         xs_validate,
                         y_validate, 
                        self.basis_0, 
                        self.basis_1)
        
        self.y_simulated = ys_mhl
        
        if metric == 'MSE':
            from sklearn.metrics import mean_squared_error
            self.error = mean_squared_error(y_validate[:,0],ys_mhl[:,0])+mean_squared_error(y_validate[:,1],ys_mhl[:,1])
            print('\n', 'MSE: %.6f '% self.error)
        

        if plot == True:
            plt.plot(xs_validate, y_validate[:, 0], 'o', color='#d73027')
            plt.plot(xs_validate, ys_mhl[:, 0], color='black')
            plt.plot(xs_validate, y_validate[:, 1], 'o', color='#fc8d59')
            plt.plot(xs_validate, ys_mhl[:, 1], color='black')
            plt.show()
        
                
                
                
                
            
            

        
        
    
         