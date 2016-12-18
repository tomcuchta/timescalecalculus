# -----
# Limitations
# -----
# at this point, a timescale can be any list, and so since it is represented by computer, it is a finite time scale (necessarily discrete)
# we are also restricted to the 1-dimensional case -- something that needs to be remedied!!!
# no complex time scale functionality yet
#
# -----
# To further generalize
# -----
# simple logic should be able to add the capability of finite unions of finite intervals and also infinite intervals
# after that, limit points like the one forced by taking the closure of {1,1/2,1/3,1/4,1/5,...} (i.e. "Cantor-Bendixson rank 1"), not sure how hard that will be
# and hopefully higher CB-rank -- if we solve some like the above, then these should be easy?
#
#
#
# Forward jump operator
# ---------------------
# the forward jump operator sigma: T->T gives the "next element" of the time scale
#
# proper time scale behavior says that inf(empty)=max(T) -- does that happen?
#
def sigma(t,timescale):
	return min([x for x in timescale if x>t]);
#
#
# Backward jump operator
# ----------------------
#
def rho(t,timescale):
	return max([x for x in timescale if x<t]);
#
#
# Forward graininess mu
# ---------------------
# 
def mu(t,timescale):
	return sigma(t,timescale)-t;
#
#
# Backward graininess nu
# ----------------------
#
def nu(t,timescale):
	return t-rho(t,timescale);
#
#
# 
