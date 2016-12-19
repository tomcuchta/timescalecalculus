# this code works for finite time scales 
#
from elementaryfunctions import *
#
# 
# delta exponential (http://timescalewiki.org/index.php/Delta_exponential)
# -----------------
# 
def dexpf(f,t,s,timescale):
	return product([1+mu(x,timescale)*f(x) for x in timescale if x >= s and x<t])
