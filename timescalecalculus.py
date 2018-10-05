import operator
from functools import reduce # Added this because in python 3.* they changed the location of the reduce() method to the functools module

#
#
# Product function from
# https://stackoverflow.com/questions/595374/whats-the-python-function-like-sum-but-for-multiplication-product
#
def product(factors):
        return reduce(operator.mul, factors, 1)


#
#
# Time scale class
#
#
class timescale:
    def __init__(self,ts,name='none'):
        self.ts = ts
        self.name = name
    #
    #
    # forward jump
    #
    #
    def sigma(self,t):
        if t==max(self.ts):
            return t
        else:
            return min([x for x in self.ts if x>t])
    #
    #
    # backwards jump
    #
    #
    def rho(self,t):
        if t==min(self.ts):
            return t
        else:
            return max([x for x in self.ts if x<t])

    #
    #
    # graininess
    #
    #
    def mu(self,t):
        return self.sigma(t)-t

    #
    #
    # backward graininess
    #
    #
    def nu(self,t):
        return t-self.rho(t)

    #
    #
    # delta derivative
    #
    #
    def dderivative(self,f,t):
        return (f(self.sigma(t))-f(t))/self.mu(t)

    #
    #
    # nabla derivative
    #
    #
    def nderivative(self,f,t):
        return (f(t)-f(self.rho(t)))/self.nu(t)

    #
    #
    # delta integral
    #
    #
    def dintegral(self,f,t,s):
        return sum([self.mu(x)*f(x) for x in self.ts if x>=s and x<t])

    #
    #
    # delta exponential
    #
    #
    def dexp_p(self,p,t,s):
#        for x in self.ts:
#            if x >= s and x < t:
#                print("x:", x)
#                print("p(x):", p(x))
#
#        print()

        return product([1+self.mu(x)*p(x) for x in self.ts if x >= s and x<t])

    #
    #
    # forward circle minus
    #
    #
    def mucircleminus(self,f,t):
        return -f(t)/(1+f(t)*self.mu(t))

    #
    #
    # The forward-derivative cosine trigonometric function.
    #
    #
    def dcos_p(self, p, t, s):
        dexp_p1 = self.dexp_p(lambda x: p(x) * 1j, t, s)

        dexp_p2 = self.dexp_p(lambda x: p(x) * -1j, t, s)

        return ((dexp_p1 + dexp_p2) / 2)

    #
    #
    # The forward-derivative sine trigonometric function.
    #
    #
    def dsin_p(self, p, t, s):
        dexp_p1 = self.dexp_p(lambda x: p(x) * 1j, t, s)

        dexp_p2 = self.dexp_p(lambda x: p(x) * -1j, t, s)

        return ((dexp_p1 - dexp_p2) / 2j)

    #
    #
    # The Laplace transform function.
    #
    #
    def laplace_transform(self, f, z, s):
        def g(t):
            return f(t) * self.dexp_p(lambda t: self.mucircleminus(z, t), self.sigma(t), s)

        return self.dintegral(g, max(self.ts), s)

#
#
# create the time scale of integers {x : a <= x <= b}
#
#
def integers(a,b):
    return timescale(list(range(a,b+1)),'integers from '+str(a)+' to '+str(b))

#
#
# create the time scale of quantum numbers of form {q^k:k=m,m+1,...,n}
# only does q^(X) where X={0,1,2,3,...} at the moment
#
def quantum(q,m,n):
    return timescale([q**k for k in range(m,n)], 'quantum numbers '+str(q)+'^'+str(m)+' to '+str(q)+'^'+str(n))


