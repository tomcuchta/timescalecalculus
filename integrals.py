from elementaryfunctions import *
# currently defined for finite time scales
#
#
# Delta integral (http://timescalewiki.org/index.php/Delta_integral)
#---------------
#
def dintegral(f,t,s,timescale):
	return sum([mu(x,timescale)*f(x) for x in timescale if x>=s and x<t]);
#
#
