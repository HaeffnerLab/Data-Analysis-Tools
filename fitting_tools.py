import numpy as np
import collections
from scipy.optimize import curve_fit


class function_fitter():
    """
    This is the parent class of specific functions which are defined below.
    This class contains the function to be fitted and its parameters, including those which are to be fitted and those which are to be held at a fixed value.
    Methods are defined to set which parameters are which, set their values, and perform the fit.

    Variables contained in this class:
    self.full_fun: Function. The general form of the function we want to fit to. Requires all input paramaters as arguments.
    self.parameters: List of strings. Lists the arguments that need to be passed to self.full_fun as parameters (this excludes the first argument, which is taken to be the x values.)
    self.reduced_fun: Function. The reduced form of full_fun, which only accepts the parameters that are not fixed as arguments.
    self.params_fixed: Dictionary. Keys are the names of parameters, corresponding to elements of self.parameters, and values are the values of the corresponding parameters. Only parameters not being fitted are included.
    self.params_tofit: List of strings. Subset of self.parameters; includes only those which are not keys of self.params_fixed.
    self.params_tofit_guesses: Dictionary. Keys are elements of self.params_tofit, values are guess values used as initial inputs for the actual fitting algorithm (scipy.optimize.curve_fit)
    self.params_tofit_fits: Dictionary. Keys are elements of self.params_tofit, values are 2-element tuples, with the first element being the fitted value and the second being the uncertainty (calculated as the square root of the corresponding element of the covariance matrix)
    self.fit_bounds: Dictionary. Keys are elements of self.params_tofit, values are 2-element tuples, with the first element being the lower bound and the second being the upper bound. +/-np.inf indicates no upper/lower bound.
    self.guesses_set: Boolean. Indicates whether guess parameters have been set yet. Is reset to False if self.set_fixed_params() is called.
    self.fit_done: Boolean. Indicates whether fit has been performed yet or not. Is reset to False if self.set_fixed_params() or self.set_guess_params() is called.
    """

    def __init__(self, function, parameters):
        """Fitting functions should generally be defined as child classes below.
        However, with this method you can pass in an arbitrary function and make it a function_fitter object,
        by passing in the function and a list of its parameters (following the same rules as the child classes below)."""
        self.full_fun = function
        self.parameters = parameters
        self.setup()

    def setup(self):
        self.reduced_fun = self.full_fun
        self.params_fixed = dict()
        self.params_tofit = self.parameters
        self.params_tofit_guesses = dict()
        self.params_tofit_fits = dict()
        self.fit_bounds = dict()
        self.guesses_set = False
        self.fit_done = False

    def print_parameters(self):
        print('Parameters:')
        for param in self.parameters: print("'"+param+"'")

    def set_fixed_params(self, paramstofix_dict):
        for param in paramstofix_dict.keys():
        # Make sure that the parameters we're trying to fix are actually parameters of the function.
            if not param in self.parameters:
                raise ValueError('"{0}" is not a listed parameter for the function {1}.'.format(param, self.__class__.__name__))
        self.guesses_set = False
        self.fit_done = False
        self.params_fixed = paramstofix_dict
        self.params_tofit = [param for param in self.parameters if param not in paramstofix_dict.keys()]
        self.params_tofit_fits = dict()
        self._set_tofit_dicts()
        self.reduced_fun = self.get_reduced_fun()

    def set_guess_params(self, paramsguess_dict):
        for param in paramsguess_dict:
            # Make sure that the parameters being set are actually parameters of the function.
            if not param in self.parameters:
                raise ValueError('"{0}" is not a listed parameter for the function {1}.'.format(param, self.__class__.__name__))
        for param in paramsguess_dict:
            # Make sure that none of the parameters set to be fixed are included here
            if param in self.params_fixed.keys():
                raise ValueError('"{}" is already set as a fixed parameter.'.format(param))
        for param in self.params_tofit:
            # Make sure that all parameters not being fixed are being given a guess value
            if param not in paramsguess_dict.keys():
                raise ValueError('"{0}" has not been set as a fixed parameter, but is not being given a guess value either! Give "{0}" a guess value or set it as a fixed parameter.'.format(param))
        self.guesses_set = True
        self.fit_done = False
        self.params_tofit_fits = dict()
        self.params_tofit_guesses = paramsguess_dict

    def set_fit_bounds(self, fitbounds_dict):
        for param in fitbounds_dict:
            # Make sure that the parameters being set are actually parameters of the function.
            if not param in self.parameters:
                raise ValueError('"{0}" is not a listed parameter for the function {1}.'.format(param, self.__class__.__name__))
        for param in fitbounds_dict:
            # Make sure that none of the parameters set to be fixed are included here
            if param in self.params_fixed.keys():
                raise ValueError('"{}" is already set as a fixed parameter.'.format(param))
        for param in fitbounds_dict:
            # Make sure the value being passed for each parameter is actually a 2-tuple
            bounds = fitbounds_dict[param]
            assert isinstance(bounds, collections.Sequence), 'Bounds must be set as a 2-tuple. Use (+ or -) np.inf to indicate no (upper or lower) bound.'
            assert len(bounds) == 2,                         'Bounds must be set as a 2-tuple. Use (+ or -) np.inf to indicate no (upper or lower) bound.'
        self.fit_done = False
        self.params_tofit_fits = dict()
        self.fit_bounds = fitbounds_dict

    def do_fit(self, x, y, yerr=None, use_qpn=False, nmeasurements=100, print_result=False):
        if not self.guesses_set:
            raise RuntimeError('Guess values have not been set for parameters to fit. Call function_fitter.set_guess_params(guess_params) with a dictionary of parameter guess values as the argument.')
        # Construct the keyword argument inputs for p0 and bounds
        p0 = self._guess_params_to_args()
        bounds = self._bounds_to_lists()
        
        # If use_qpn is true, the quantum-projection-noise-calculated uncertainty is used.
        # For excitation p from N measurements, this is given by either sqrt(p*(1-p)/N) or 1/(N+2), whichever is larger [Thomas Monz thesis, Innsbruck]
        if use_qpn==True:
            yerr = np.maximum(np.sqrt(y*(1.0-y)/nmeasurements), 1.0/(nmeasurements+2))
        elif use_qpn=='parity':
            p = (y+1)/2.0
            yerr = np.maximum(np.sqrt(p*(1.0-p)/nmeasurements), 2.0/(nmeasurements+2))
        
        # absolute_sigma should be used in curve_fit only if a y error has been specified.
        abs_sig = True if yerr is not None else False

        # Get the fitted values and uncertainties
        (fitvals, covar) = curve_fit(self.reduced_fun, x, y, sigma=yerr, p0=p0, bounds=bounds, absolute_sigma=abs_sig)
        
        # Write the fitted values and uncertainties to the dictionary self.params_tofit_fits
        for (i, param) in enumerate(self.params_tofit):
            self.params_tofit_fits[param] = (fitvals[i], np.sqrt(covar[i, i]))
        
        # Write residuals and rsq to self.residuals and self.rsq
        self.residuals = np.array(y)-self.reduced_fun(np.array(x),*fitvals)
        ss_r = np.sum(self.residuals**2)
        ss_tot = np.sum((np.array(y)-np.mean(y))**2)
        self.rsq = 1-(ss_r/ss_tot)
        
        self.fit_done = True
        if print_result:
            self.print_fits()
    
    def eval_with_guesses(self, x):
        if not self.guesses_set:
            raise RuntimeError('Guess values have not been set for parameters to fit. Call function_fitter.set_guess_params(guess_params) with a dictionary of parameter guess values as the argument.')
        reduced_args = self._guess_params_to_args()
        return self.reduced_fun(x, *reduced_args)

    def eval_with_fits(self, x):
        if not self.fit_done:
            raise RuntimeError('Need to perform a fit first! Call function_fitter.do_fit().')
        reduced_args = self._fit_params_to_args()
        return self.reduced_fun(x, *reduced_args)
    
    def get_status(self):
        return {'guesses_set':self.guesses_set, 'fit_done':self.fit_done}
    
    def get_fixedparams(self):
        return self.params_fixed.copy()

    def get_guesses(self):
        if not self.guesses_set:
            raise RuntimeError('Guess values have not been set for parameters to fit. Call function_fitter.set_guess_params(guess_params) with a dictionary of parameter guess values as the argument.')
        return self.params_tofit_guesses.copy()
    
    def get_fits(self):
        # Returns self.params_tofit_fits, a dictionary of fitted values and uncertainties
        # Each key is the name of a fitted parameter, and each value is a 2-tuple of the format (fitvalue, uncertainty)
        if not self.fit_done:
            raise RuntimeError('Need to perform a fit first! Call function_fitter.do_fit().')
        return self.params_tofit_fits.copy()

    def get_reduced_fun(self):
        # Returns a function that accepts a reduced set of input arguments, i.e. with the fixed parameters set to their specified values (as defined by the dictionary self.params_fixed)
        def reduced_fun(x, *reduced_args):
            reduced_args = list(reduced_args)
            full_args = [None]*len(self.parameters)
            for (i, param) in enumerate(self.parameters):
                if param in self.params_tofit:
                    full_args[i] = reduced_args.pop(0)
                else:
                    full_args[i] = self.params_fixed[param]
            return self.full_fun(x, *full_args)
        return reduced_fun

    def print_fits(self):
        # Print out the fitted values and uncertainties
        if not self.fit_done:
            raise RuntimeError('Need to perform a fit first! Call function_fitter.do_fit().')
        for param in self.params_tofit_fits:
            print('{0}: {1} +- {2}'.format(param, self.params_tofit_fits[param][0], self.params_tofit_fits[param][1]))

    def _set_tofit_dicts(self):
        # Called after fixed parameters have been set.
        # Initializes the dictionaries params_tofit_guesses and params_tofit_fits with the correct keys (i.e. the parameters to be fitted), and None values.
        params_tofit_guesses = dict()
        params_tofit_fits = dict()
        for param in self.params_tofit:
            params_tofit_guesses[param] = None
            params_tofit_fits[param] = None

    def _guess_params_to_args(self):
        # Takes the information stored in params_tofit_guesses and turns it into a list of fit values ordered by the relevent parameters' ordering in self.parameters
        reduced_args = [None]*len(self.params_tofit)
        for (i, param) in enumerate(self.params_tofit):
            reduced_args[i] = self.params_tofit_guesses[param]
        return reduced_args

    def _fit_params_to_args(self):
        # Takes the information stored in params_tofit_fits and turns it into a list of fit values ordered by the relevent parameters' ordering in self.parameters
        reduced_args = [None]*len(self.params_tofit)
        for (i, param) in enumerate(self.params_tofit):
            reduced_args[i] = self.params_tofit_fits[param][0]
        return reduced_args

    def _bounds_to_lists(self):
        # Turns the fitting bounds set in the dictionary self.fit_bounds into a form which can be passed into scipy.optimize.curve_fit's "bounds" keyword argument
        lbounds = [None]*len(self.params_tofit)
        ubounds = [None]*len(self.params_tofit)
        for (i, param) in enumerate(self.params_tofit):
            if param in self.fit_bounds.keys():
                lbounds[i] = self.fit_bounds[param][0]
                ubounds[i] = self.fit_bounds[param][1]
            else:
                lbounds[i] = -np.inf
                ubounds[i] = np.inf
        return (lbounds, ubounds)


def print_functions():
    # Print out a list of the fitting functions which have been defined
    import sys, inspect
    clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    print('Options:')
    for (name, obj) in clsmembers:
        if name != 'function_fitter':
            print(name+'()')


from fitting_functions_general import *
from fitting_functions_special import *
