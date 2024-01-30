import timescalecalculus as tsc
import scipy.optimize as so
import matplotlib.pyplot as plt
import numpy as np
#
# define the time scale
#
ts=tsc.timescale([0,1,2,3,3.5,4,4.5,5,5.5,6,6.5,7,8,9,10,12,14,14.5,15,15.5,16,16.5,17,19,21,23,23.5,24,24.5,25,25.5,26,28,29,29.5,29.6,30])
#
# define boundary values
#
initialtvalue = 0
finaltvalue = 30
initialoutputvalue = 0
finaloutputvalue = 1
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
    return 0.00072*(kappa0(t,alpha)*(-6+ts.sigma(t)/10)-kappa1(t,alpha)*(-6+t/10)-kappa0(t,alpha)/10)
def p(t):
    return -6+t/10
def f(t):
    return 0
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
alpha=0.86
initvalue=so.broyden1(minimizethis,initialoutputvalue)
ts.plot(soln,color='orange',label=r'$\alpha=$'+str(alpha),zorder=2)
plt.plot(ts.ts,[soln(t) for t in ts.ts],color='orange',alpha=0.2)
#
# plot again with a different alpha
#
alpha=0.35
initvalue=so.broyden1(minimizethis,initialoutputvalue)
ts.plot(soln,color='blue',label=r'$\alpha=$'+str(alpha),zorder=2)
plt.plot(ts.ts,[soln(t) for t in ts.ts],color='blue',alpha=0.2)
#
# plot again with a different alpha
#
alpha=0.1
initvalue=so.broyden1(minimizethis,initialoutputvalue)
ts.plot(soln,color='red',label=r'$\alpha=$'+str(alpha),zorder=2)
plt.plot(ts.ts,[soln(t) for t in ts.ts],color='red',alpha=0.2)
#
# label the x axis
#
plt.xlabel(r"$t$")
#
# label the y axis
#
plt.ylabel(r"$u(t)$")
#
# generate the legend
#
ts.plt.legend()
#
# save the figure with high dpi for rendering in the paper
#
plt.savefig('example5.png',dpi=500)
