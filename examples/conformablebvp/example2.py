import timescalecalculus as tsc
import scipy.optimize as so
import matplotlib.pyplot as plt
import numpy as np
#ts=tsc.integers(0,5)
ts=tsc.timescale([0, 0.5, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.5, 4])#2.2, 2.3,#2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.5, 4.0])
#ts=tsc.timescale([0,1,1.2,1.3,4,4.5,4.7,5,5.1,6,7,8.2,9.9,10,10.1,10.2,10.3,14,15])
#ts=tsc.timescale([0,1,2,[4,5],6,7,8,[9,10],11,12,[13,14],15])
#ts=tsc.timescale([0,[1,2],3,[4,5],6,[7,8],9,[10,11],12,[13,14],15])
#ts=tsc.timescale([0,[1,2],3,[4,5],6,[7,8],9,[10,11],12,[13,14],15,[16,17],18,[19,20],21,[22,23],24,[25,26],27,[28,29],30,[31,32],33,[34,35],36,[37,38],39,[40,41],42,[43,44],45,[46,47],48,[49,50],51])
#ts=tsc.timescale([0,1,1.5,2,3,3.5,4,5,5.5,6,7,7.5,8,9,9.5,10,11,11.5,12,13,13.5,14,15,15.5,16,17,17.5,18,19,19.5,20,21,21.5,22,23,23.5,24,25,25.5,26,27,27.5,28,29,29.5,30,31,31.5])

initialtvalue = 0
finaltvalue = 4
initialoutputvalue = 250
finaloutputvalue = 250

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

alpha=1.0
#
# Suppose we require boundary values as y(0)=5 and y(15)=10
#
def A(t):
    return (p(t)*kappa1(t,alpha)*kappa0(t,alpha)+kappa0(t,alpha)*kappa1(t,alpha)*p(ts.sigma(t))+kappa0(t,alpha)*kappa0(t,alpha)*ts.dderivative(lambda x: p(x),t)+q(t)*ts.mu(t))/(kappa0(t,alpha)*p(ts.sigma(t)))

def B(t):
    return (p(t)*kappa1(t,alpha)*kappa1(t,alpha)+kappa0(t,alpha)*kappa1(t,alpha)*ts.dderivative(lambda x: p(x),t)+q(t))/(kappa0(t,alpha)*p(ts.sigma(t)))
   
def y_prime_vector(vector,t):
    x1, x2= vector
    dt_vector=[x2, -A(t)*x1-B(t)*x2+f(t)/(kappa0(t,alpha)*p(ts.sigma(t)))]
    return dt_vector

#def soln(t):
#    return ts.solve_ode_system_for_t([1,-3], 0, t, y_prime_vector)[0]

# the following function evaluates the solution at t=15 and subtracts the boundary value we want
def minimizethis(a):
    return abs(ts.solve_ode_system_for_t([initialoutputvalue,a],0,finaltvalue,y_prime_vector)[0]-finaloutputvalue)

# now we minimize that function to find the value of a that causes the minimimum -- this will yield an initial value for y'(0) that 
# gives us the boundary value we are looking for
#print("to get y(15)=10, take initial condition y'(0)="+str(so.broyden1(minimizethis,5)))    

initvalue=so.newton_krylov(minimizethis,initialoutputvalue)
def soln(t):
    return ts.solve_ode_system_for_t([initialoutputvalue,initvalue], 0, t, y_prime_vector)[0]

plt.scatter([initialtvalue,finaltvalue],[initialoutputvalue,finaloutputvalue],marker='X',color='black',zorder=3,s=50)

ts.plot(soln,color='black',label=r'$\alpha=$'+str(alpha),zorder=2)
plt.plot(ts.ts,[soln(t) for t in ts.ts],color='black',alpha=0.2)
alpha=0.85
initvalue=so.broyden1(minimizethis,initialoutputvalue)
ts.plot(soln,color='orange',label=r'$\alpha=$'+str(alpha),zorder=2)
plt.plot(ts.ts,[soln(t) for t in ts.ts],color='orange',alpha=0.2)
alpha=0.77
initvalue=so.broyden1(minimizethis,initialoutputvalue)
ts.plot(soln,color='blue',label=r'$\alpha=$'+str(alpha),zorder=2)
plt.plot(ts.ts,[soln(t) for t in ts.ts],color='blue',alpha=0.2)
alpha=0.35
initvalue=so.broyden1(minimizethis,initialoutputvalue)
ts.plot(soln,color='red',label=r'$\alpha=$'+str(alpha),zorder=2)
plt.plot(ts.ts,[soln(t) for t in ts.ts],color='red',alpha=0.2)


plt.xlabel("Sensor location (in)")
plt.ylabel("Temperature reading (F)")

ts.plt.legend()
plt.savefig('coolingproblem.png',dpi=500)
