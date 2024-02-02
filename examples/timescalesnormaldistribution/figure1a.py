#
# created 2021.08.26 by tom cuchta
# used in published manuscript
# Umit Aksoy, Tom Cuchta, Svetlin Georgiev, and Yeliz Okur. A normal distribution on time scales with application. Filomat, 36(16):5391-5404, 2022.
# https://www.pmf.ni.ac.rs/filomat-content/2022/36-16/36-16-4-17039.pdf
#
import timescalecalculus as tsc
import matplotlib.pyplot as plt

ts=tsc.timescale([-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7])
def w1(t,m,p):
    return (t-m)/(p*p)

def w2(t,m,p):
    return -w1(t,m,p)

def q(t,m,p):
    if t>=m:
        return ts.mucircleminus(lambda t: w1(t,m,p),t)
    if t<m:
        return ts.mucircleminus(lambda t: w2(t,m,p),t)

def N1(t,m,p):
    if t>=m:
        return ts.dexp_p(lambda t: q(t,m,p),ts.sigma(t),m)
    elif t<m:
        return ts.dexp_p(lambda t: q(t,m,p),m,t)

ts.plot(lambda t: N1(t,0,0.5),label=r'$p=0.5$',marker='o',markersize=4)
ts.plot(lambda t: N1(t,0,1),color='red',label=r'$p=1$',marker='X',markersize=5)
ts.plot(lambda t: N1(t,0,2),color='green',label=r'$p=2$',marker='*',markersize=6)
plt.legend()
plt.savefig('figure1a.png',dpi=1000)
