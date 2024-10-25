import numpy as np
import scipy.constants as scc
import scipy.special as sp
import scipy.misc as scm
from scipy.special import eval_genlaguerre as laguerre

from fitting_tools import function_fitter


"""
Below are the functions which have been defined, as child classes of the class function_fitter.

Defining a new fitting function is simple! (I hope.)

Short version:
Just copy one of the classes already defined. It may be clear what needs to be changed given the examples below. If not...
Only change (1) self.parameters and (2) the definition of full_fun() (which is the function you want to fit to).
The first argument to full_fun() after "self" should be the x-axis values. Make sure the elements of the list self.parameters match all the arguments after this one.

Long version:
To define a function to be used as a fitting function, define the class as follows:
A. Define the class as a child instance of the class function_fitter.
B. __init__ should consist of 2 lines:
   1. self.parameters = (a list of strings, see step C for details)
   2. self.setup()
C. Define a function full_fun(self, ...) to be the function to be fitted. The first argument after self should be the x-axis, and the rest are just any arguments the function needs.
   These will each later be set to either be fitted or fixed at some defined value.
   The variable self.parameters should be a list of strings denoting these parameters (the ones excluding the x-axis), so the length of this list should be equal to the number of arguments
   excluding the first one. These don't necessarily need to have the same name as the arguments themselves (though it's probably easier that way), but the arguments they refer to *DO* need to
   be in the same order.
"""


class line(function_fitter):
    def __init__(self):
        self.parameters = ['m', 'b']
        self.setup()
        
    def full_fun(self, x, m, b):
        return m*x + b

class power_law(function_fitter):
    def __init__(self):
        self.parameters = ['a', 'b']
        self.setup()
        
    def full_fun(self, x, a, b):
        return a*x**b

class gaussian(function_fitter):
    #defined as in grapher
    def __init__(self):
        self.parameters = ['center', 'scale', 'sigma', 'offset']
        self.setup()
        
    def full_fun(self, x, center, scale, sigma, offset):
        return offset + scale*np.exp(-(x-center)**2 / (2*sigma**2))

class lorentzian(function_fitter):
    #defined as in grapher
    def __init__(self):
        self.parameters = ['center', 'scale', 'fwhm', 'offset']
        self.setup()
        
    def full_fun(self, x, center, scale, fwhm, offset):
        return offset + scale*0.5*fwhm/((x-center)**2.0 + fwhm**2.0/4.0)
        
class exponential_decay(function_fitter):
    def __init__(self):
        self.parameters = ['tau', 'startfrom', 'decayto']
        self.setup()

    def full_fun(self, times, tau, startfrom, decayto):
        return startfrom + (decayto-startfrom)*(1-np.exp(-times/tau))
    
class gaussian_decay(function_fitter):
    def __init__(self):
        self.parameters = ['tau', 'startfrom', 'decayto']
        self.setup()

    def full_fun(self, times, tau, startfrom, decayto):
        return startfrom + (decayto-startfrom)*(1-np.exp(-times**2/(2*tau**2)))

class ramsey_decay(function_fitter):
    # Simple exponential decay with a sinusoudal oscillation
    def __init__(self):
        self.parameters = ['freq', 'tau_us', 'start_from', 'decay_to']
        self.setup()
        
    def full_fun(self, times_us, freq, tau_us, start_from, decay_to):
        #convert inputs to SI
        times = 1e-6 * times_us
        w = 2*np.pi*freq
        tau = 1e-6 * tau_us
        
        return (start_from-decay_to)*np.exp(-times/tau)*np.cos(w*times) + decay_to

class phase_scan(function_fitter):
    def __init__(self):
        self.parameters = ['contrast', 'phi0', 'offset']
        self.setup()

    def full_fun(self, phi, contrast, phi0, offset):
        phi_rad = phi*np.pi/180.0
        phi0_rad = phi0*np.pi/180.0
        return 0.5 + contrast/2.0*np.sin(phi_rad-phi0_rad) + offset

class parity_scan(function_fitter):
    def __init__(self):
        self.parameters = ['contrast', 'phi0', 'offset']
        self.setup()

    def full_fun(self, phi, contrast, phi0, offset):
        phi_rad = phi*np.pi/180.0
        phi0_rad = phi0*np.pi/180.0
        return contrast/2.0*np.sin(2*(phi_rad-phi0_rad)) + offset

class cosine(function_fitter):
    def __init__(self):
        self.parameters = ['angfreq', 'amplitude', 'offset', 'phase_deg']
        self.setup()

    def full_fun(self, x, angfreq, amplitude, offset, phase_deg):
        return offset + amplitude*np.cos(angfreq*x + np.pi/180*phase_deg)

class rabi_flop_thermal(function_fitter):
    # Supports up to second order sidebands
    def __init__(self):
        self.parameters = ['Omega_kHz', 'delta_kHz', 'eta', 'sideband_order', 'scale', 'nbar', 'turnon_delay_us']
        self.setup()

    def full_fun(self, times_us, Omega_kHz, delta_kHz, eta, sideband_order, scale, nbar, turnon_delay_us):
        times = 1e-6 * np.array([(t if t>=0 else 0) for t in (times_us-turnon_delay_us)])
        Omega = 1e3 * 2*np.pi * Omega_kHz
        delta = 1e3 * 2*np.pi * delta_kHz

        nmax = 1000
        ns = np.arange(nmax)

        omega_n = Omega*self._compute_rabi_coupling(eta, sideband_order, ns)
        p_n = 1.0/ (nbar + 1.0) * (nbar / (nbar + 1.0))**ns
        exc_n = np.outer(p_n * omega_n**2/(omega_n**2+delta**2), np.ones_like(times)) * np.sin(np.outer(np.sqrt(omega_n**2 + delta**2), times/2.0))**2
        exc = scale * np.sum(exc_n, axis = 0)

        return exc

    def _compute_rabi_coupling(self, eta, sideband_order, ns):
        if sideband_order == 0:
            coupling_func = lambda n: np.exp(-1./2*eta**2) * laguerre(n, 0, eta**2)
        elif sideband_order == 1:
            coupling_func = lambda n: np.exp(-1./2*eta**2) * eta**(1)*(1./(n+1.))**0.5 * laguerre(n, 1, eta**2)
        elif sideband_order == 2:
            coupling_func = lambda n: np.exp(-1./2*eta**2) * eta**(2)*(1./((n+1.)*(n+2)))**0.5 * laguerre(n, 2, eta**2)
        elif sideband_order == -1:
            coupling_func = lambda n: 0 if n == 0 else np.exp(-1./2*eta**2) * eta**(1)*(1./(n))**0.5 * laguerre(n - 1, 1, eta**2)
        elif sideband_order == -2:
            coupling_func = lambda n: 0 if n <= 1 else np.exp(-1./2*eta**2) * eta**(2)*(1./((n)*(n-1.)))**0.5 * laguerre(n - 2, 2, eta**2)
        return np.array([coupling_func(n) for n in ns])

class poissonian(function_fitter):
    def __init__(self):
        # mu is center, k is value, k has to be positive
        self.parameters = ['mu']
        self.setup()
    
    def full_fun(self, k, mu):
        return np.exp(-mu)*mu**k/scm.factorial(k)

class poissonian2(function_fitter):
    # two poisson mass functions for fitting to readout histograms. 
    # (assuming no relevant D state decay)
    def __init__(self):
        # mu is center, k is value, k has to be positive
        self.parameters = ['mu1', 'mu2']
        self.setup()
    
    def full_fun(self,k,mu1,mu2):
        return np.exp(-mu1)*mu1**k/scm.factorial(k) + np.exp(-mu2)*mu2**k/scm.factorial(k)
    
class spectrum_2level(function_fitter):
    # Spectrum of an ideal 2-level transition
    def __init__(self):
        self.parameters = ['f0_MHz', 'Omega_kHz', 'time_us', 'scale']
        self.setup()
        
    def full_fun(self, f_MHz, f0_MHz, Omega_kHz, time_us, scale):
        delta = 1e6 * 2*np.pi * (f_MHz-f0_MHz)
        Omega = 1e3 * 2*np.pi * Omega_kHz
        time = 1e-6 * time_us
        return scale * Omega**2/(Omega**2+delta**2) * np.sin(np.sqrt(Omega**2+delta**2)*time/2)**2

class spectrum(function_fitter):
    # Spectrum whose value at delta = 0 is not defined by the value of sin^2(Omega*t/2) but rather directly by the parameter 'scale'
    def __init__(self):
        self.parameters = ['f0_MHz', 'Omega_kHz', 'scale']
        self.setup()
        
    def full_fun(self, f, f0_MHz, Omega_kHz, scale):
        delta = 1e6 * 2*np.pi * (f-f0_MHz)
        Omega = 1e3 * 2*np.pi * Omega_kHz
        time = np.pi / Omega
        return scale * Omega**2/(Omega**2+delta**2) * np.sin(np.sqrt(Omega**2+delta**2)*time/2)**2

class sinc_squared(function_fitter):
    def __init__(self):
        self.parameters = ['x0', 'width', 'scale']
        self.setup()
        
    def full_fun(self, x, x0, width, scale):
        return scale*np.sinc((x-x0)/width)**2