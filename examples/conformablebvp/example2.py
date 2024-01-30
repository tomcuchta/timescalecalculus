import timescalecalculus as tsc
import scipy.optimize as so
import matplotlib.pyplot as plt
import numpy as np
#
# define the time scale
#
ts=tsc.timescale([0, 0.5, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.5, 4])#2.2, 2.3,#2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.5, 4.0])
#
# define boundary values
#
initialtvalue = 0
finaltvalue = 4
initialoutputvalue = 250
finaloutputvalue = 250
#
# choose an initial value of alpha
#
alpha=1.0

#
# define the functions needed for expressing the problem
#
def kappa0(t,alpha):
    return alpha

def kappa1(t,alpha):
    return 1-alpha

def q(t):
    return 5.00156

def p(t):
    return -t
#    return 1/(1+t)

def f(t):
    return 373.117


#
# coefficient of x^Del=x2
#
def A(t):
    return (p(t)*kappa1(t,alpha)*kappa0(t,alpha)+kappa0(t,alpha)*kappa1(t,alpha)*p(ts.sigma(t))+kappa0(t,alpha)*kappa0(t,alpha)*ts.dderivative(lambda x: p(x),t)+q(t)*ts.mu(t))/(kappa0(t,alpha)*p(ts.sigma(t)))
#
# coefficient of x=x1
#
def B(t):
    return (p(t)*kappa1(t,alpha)*kappa1(t,alpha)+kappa0(t,alpha)*kappa1(t,alpha)*ts.dderivative(lambda x: p(x),t)+q(t))/(kappa0(t,alpha)*p(ts.sigma(t)))

#
# set up the system
#
def y_prime_vector(vector,t):
    x1, x2= vector
    dt_vector=[x2, -A(t)*x2-B(t)*x1+f(t)/(kappa0(t,alpha)*p(ts.sigma(t)))]
    return dt_vector
#
# the following function evaluates the solution at t=finaltvalue and subtracts the boundary value finaloutputvalue
# the independent variable a controls the value of x'(initialtvalue) -- later we choose a to minimize it
# this is called the "shooting method"
# https://en.wikipedia.org/wiki/Shooting_method
#
def minimizethis(a):
    return abs(ts.solve_ode_system_for_t([initialoutputvalue,a],0,finaltvalue,y_prime_vector)[0]-finaloutputvalue)
#
# now we minimize that function to find the value of a that causes the minimimum -- this will yield an initial value for y'(0) that 
# gives us the boundary value we are looking for -- we name that initialvalue inintvalue
#
initvalue=so.newton_krylov(minimizethis,initialoutputvalue)
#
# use initvalue to solve the system
# remember initvalue was chosen so that the right-side boundary value will be satisfied
#
def soln(t):
    return ts.solve_ode_system_for_t([initialoutputvalue,initvalue], 0, t, y_prime_vector)[0]
#
# scatter plot for the boundary points, marked with an "X"
# followed by a plot on the time scale of the solution
# followed by a light plot that connects the dots of the time scale plot
#
plt.scatter([initialtvalue,finaltvalue],[initialoutputvalue,finaloutputvalue],marker='X',color='black',zorder=3,s=50)
ts.plot(soln,color='black',label=r'$\alpha=$'+str(alpha),zorder=2)
plt.plot(ts.ts,[soln(t) for t in ts.ts],color='black',alpha=0.2)

#
# plot again with a different alpha
#
alpha=0.85
initvalue=so.broyden1(minimizethis,initialoutputvalue)
ts.plot(soln,color='orange',label=r'$\alpha=$'+str(alpha),zorder=2)
plt.plot(ts.ts,[soln(t) for t in ts.ts],color='orange',alpha=0.2)
#
# plot again with a different alpha
#
alpha=0.77
initvalue=so.broyden1(minimizethis,initialoutputvalue)
ts.plot(soln,color='blue',label=r'$\alpha=$'+str(alpha),zorder=2)
plt.plot(ts.ts,[soln(t) for t in ts.ts],color='blue',alpha=0.2)
#
# plot again with a different alpha
#
alpha=0.35
initvalue=so.broyden1(minimizethis,initialoutputvalue)
ts.plot(soln,color='red',label=r'$\alpha=$'+str(alpha),zorder=2)
plt.plot(ts.ts,[soln(t) for t in ts.ts],color='red',alpha=0.2)

#
# label the x axis
#
plt.xlabel("Sensor location (in)")
#
# label the y axis
#
plt.ylabel("Temperature reading (F)")
#
# generate the legend
#
ts.plt.legend()
#
# save the figure with high dpi for rendering in the paper
#
plt.savefig('coolingproblem.png',dpi=500)
