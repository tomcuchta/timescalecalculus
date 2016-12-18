from elementaryfunctions import *
# currently defined for finite time scales
#
#
# Delta derivative (http://timescalewiki.org/index.php/Delta_derivative)
# ----------------
#
def dderivative(f,t,ts):
	return (f(sigma(t,ts))-f(t))/mu(t,ts);
#
#
# Nabla derivative (http://timescalewiki.org/index.php/Nabla_derivative)
# ----------------
#
def nderivative(f,t,ts):
	return (f(t)-f(rho(t,ts)))/nu(t,ts);

