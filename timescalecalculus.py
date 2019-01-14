import operator
from functools import reduce # Added this because in python 3.* they changed the location of the reduce() method to the functools module
from scipy import integrate
from scipy.misc import derivative
import numpy as np
import matplotlib.pyplot as plt

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

        # The following two dictionary data members are used to for the memoization of the g_k and h_k functions of this class.
        self.memo_g_k = {}
        self.memo_h_k = {}

        self.g_k_callCount = 0 # Temporary data member used to test memoization of g_k function.
        self.h_k_callCount = 0 # Temporary data member used to test memoization of h_k function.
        self.g_k_not_memoized_callCount = 0 # Temporary data member used to test the g_k_not_memoized function.

        #
        # The following code validates the user-specified timescale to ensure that there are no overlaps such as:
        #   - a point given more than once
        #   - a point that is included in an interval
        #   - a point that is the starting value or ending value of an interval
        #   - an interval given more than once
        #   - overlapping intervals
        #
        # If a timescale is detected as invalid an Exception will be generated with a corresponding message that describes the cause of the invalidity.
        #
        for listItem in ts:
            if isinstance(listItem, list):
                if len(listItem) > 2:
                    raise Exception("Invalid timescale declaration: you cannot have an interval with more than one starting and one ending value.")

                if len(listItem) < 2:
                    raise Exception("Invalid timescale declaration: an interval must have a starting value and an ending value.")

                if listItem[0] > listItem[1]:
                    raise Exception("Invalid timescale declaration: you cannot have an interval in which the ending value is smaller than the starting value.")

                if listItem[0] == listItem[1]:
                    raise Exception("Invalid timescale declaration: you cannot have an interval in which the starting value and ending value are equal (such an interval should be declared as a point).")

            for listItemToCompare in ts:
                if listItem == listItemToCompare and listItem is not listItemToCompare:
                    raise Exception("Invalid timescale declaration: you cannot include the same point or interval more than once.")

                if listItem is not listItemToCompare:
                    if isinstance(listItem, list) and isinstance(listItemToCompare, list):
                        if (listItem[0] >= listItemToCompare[0] and listItem[0] <= listItemToCompare[1]) or (listItem[1] >= listItemToCompare[0] and listItem[1] <= listItemToCompare[1]):
                            raise Exception("Invalid timescale declaration: you cannot have overlapping intervals.")

                    if isinstance(listItem, list) and not isinstance(listItemToCompare, list):
                        if listItemToCompare >= listItem[0] and listItemToCompare <= listItem[1]:
                            raise Exception("Invalid timescale declaration: you cannot declare a point that is included in an interval (you cannot declare a value more than once).")

                    if not isinstance(listItem, list) and isinstance(listItemToCompare, list):
                        if listItem >= listItemToCompare[0] and listItem <= listItemToCompare[1]:
                            raise Exception("Invalid timescale declaration: you cannot declare a point that is included in an interval (you cannot declare a value more than once).")

        print("Timescale successfully constructed:")
        print("Timescale:", self.ts)
        print("Timescale name:", self.name)

    #
    #
    # forward jump
    #
    #
    def sigma(self,t):
        tIndex = 0
        tNext = None
        iterations = 0

        for x in self.ts:
            if (not isinstance(x, list) and t == x) or (isinstance(x, list) and t >= x[0] and t <= x[1]):
                tIndex = iterations
                break

            iterations = iterations + 1

        if (tIndex + 1) == len(self.ts):
            return t

        elif isinstance(self.ts[tIndex], list):
            if (t != self.ts[tIndex][1]):
                return t

            else:
                if (isinstance(self.ts[tIndex + 1], list)):
                    return self.ts[tIndex + 1][0]

                else:
                    return self.ts[tIndex + 1]

        else:
            tNext = self.ts[tIndex + 1]

            if isinstance(tNext, list):
                return tNext[0]

            else:
                return tNext

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
        for x in self.ts:
            if isinstance(x, list) and t >= x[0] and t < x[1]:
                return 0

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
        if self.sigma(t) == t:
            return derivative(f, t, dx=(1.0/2**16))

        else:
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
        # The following code checks that t and s are elements of the timescale

        tIsAnElement = False
        sIsAnElement = False

        for x in self.ts:
            if not isinstance(x, list) and x == t:
                tIsAnElement = True

            if not isinstance(x, list) and x == s:
                sIsAnElement = True

            if isinstance(x, list) and  (t >= x[0] and t <= x[1]):
                tIsAnElement = True

            if isinstance(x, list) and  (s >= x[0] and s <= x[1]):
                sIsAnElement = True

            if tIsAnElement and sIsAnElement:
                break

        if not tIsAnElement and not sIsAnElement:
            raise Exception("The bounds of the dintegral function, t and s, are not elements of the timescale.")

        elif not tIsAnElement:
            raise Exception("The upper bound of dintegral function, t, is not an element of timescale.")

        elif not sIsAnElement:
            raise Exception("The lower bound of dintegral function, s, is not an element of timescale.")


        # Validation code ends

        points = []
        intervals = []

        for x in self.ts:
            if not isinstance(x, list) and s <= x and t > x:
                points.append(x)

            elif isinstance(x, list) and s <= x[0] and t > x[1]:
                points.append(x[1])
                intervals.append(x)

            elif isinstance(x, list) and s <= x[0] and t == x[1]:
                intervals.append(x)

            elif isinstance(x, list) and (s >= x[0] and s <= x[1]) and (t > x[1]):
                points.append(x[1])
                intervals.append([s, x[1]])

            elif isinstance(x, list) and (s >= x[0] and s <= x[1]) and (t == x[1]):
                intervals.append([s, x[1]])

            elif isinstance(x, list) and (s >= x[0] and s < x[1]) and (t < x[1]):
                intervals.append([s, t])

            elif isinstance(x, list) and (s < x[0]) and (t >= x[0] and t < x[1]):
                intervals.append([x[0], t])

        sumOfIntegratedPoints = sum([self.mu(x)*f(x) for x in points])

        sumOfIntegratedIntervals = sum([integrate.quad(f, x[0], x[1])[0] for x in intervals])

        return sum([sumOfIntegratedPoints, sumOfIntegratedIntervals])

    #
    #
    # Generalized g_k polynomial from page 38 with memoization.
    #
    #
    def g_k(self, k, t, s):
        self.g_k_callCount = self.g_k_callCount + 1

        if (k < 0):
            raise Exception("k should never be less than 0!")

        elif (k != 0):
            currentKey = str(k) + ":" + str(t) + ":" + str(s)

            if currentKey in self.memo_g_k:
#                print("found key =", currentKey, "with value =", self.memo_g_k[currentKey])

                return self.memo_g_k[currentKey]

            else:
                def g(x):
                   return self.g_k(k - 1, self.sigma(x), s)

                integralResult = self.dintegral(g, t, s)

                self.memo_g_k[currentKey] = integralResult

#                print("computed integral =", integralResult, "and created key =", currentKey, "with value =", integralResult)

                return integralResult

        elif (k == 0):
#            print("***** g_k : k == 0 *****")

            return 1

    #
    #
    # Not memoized version of the g_k function of this class. Used for testing.
    #
    #
    def g_k_not_memoized(self, k, t, s):
        self.g_k_not_memoized_callCount = self.g_k_not_memoized_callCount + 1

        if (k < 0):
            raise Exception("k should never be less than 0!")

        elif (k != 0):
           def g(x):
               return self.g_k_not_memoized(k - 1, self.sigma(x), s)

           return self.dintegral(g, t, s)

        elif (k == 0):
            return 1

    #
    #
    # Generalized h_k polynomial from page 38 with memoization.
    #
    #
    def h_k(self, k, t, s):
        self.h_k_callCount = self.h_k_callCount + 1

        if (k < 0):
            raise Exception("k should never be less than 0!")

        elif (k != 0):
            currentKey = str(k) + ":" + str(t) + ":" + str(s)

            if currentKey in self.memo_h_k:
#                print("found key =", currentKey, "with value =", self.memo_h_k[currentKey])

                return self.memo_h_k[currentKey]

            else:
                def h(x):
                    return self.h_k(k - 1, x, s)

                integralResult = self.dintegral(h, t, s)

                self.memo_h_k[currentKey] = integralResult

#                print("computed integral =", integralResult, "and created key =", currentKey, "with value =", integralResult)

                return integralResult

        elif (k == 0):
#            print("***** h_k : k == 0 *****")

            return 1

    #
    #
    # delta exponential
    #
    #
    def dexp_p(self,p,t,s):
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
    # Plotting functionality.
    #
    # Required arguments:
    #  f:
    #  The function that will determine the y values of the graph - the x values are determined by the current timescale values.
    #
    #  stepSize:
    #  The accuracy to which the intervals are drawn in the graph - the smaller the value, the higher the accuracy and overhead.
    #
    # Optional arguments:
    #  discreteStyle and intervalStyle:
    #  These arguments determine the color, marker, and line styles of the graph.
    #  They accept string arguments of 3-part character combinations that represent a color, marker style, and line style.
    #  For instance the string "-r." indicates that the current (x, y) points should be plotted with a connected line
    #  (represented by the "-"), in a red color (represented by the "r"), and with a point marker (represented by the ".").
    #  These character combinations can by in any order UNLESS the reordering changes the interpretation of the character combination.
    #  For instance, the string "r-." does not produce the same result as the string "-r.".
    #  This is because "-." indicates a "dash-dot line style" and is therefore no longer interpreted as a "point marker" with a "solid line style".
    #  See the notes section of this resource for more information: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
    #
    #  markerSize:
    #  Determines the size of the any markers used in the graph.
    #
    #  lineWidth:
    #  Determines the width of any lines in the graph.
    #
    #
    def plot(self, f, stepSize, discreteStyle='b.', intervalStyle='r-', markerSize=4, lineWidth=2):
        # Testing code start
        print("discreteStyle =", discreteStyle)
        print("intervalStyle =", intervalStyle)
        print("markerSize =", markerSize)
        print("lineWidth =", lineWidth)
        print("\n-----------------------------------\n")
        # Testing code end

        xDiscretePoints = []
        yDiscretePoints = []

        intervals = []

        for tsItem in self.ts:
            if isinstance(tsItem, list):
                xIntervalPoints = []
                yIntervalPoints = []

                for intervalValue in np.arange(tsItem[0], tsItem[1], stepSize):
                    xIntervalPoints.append(intervalValue)
                    yIntervalPoints.append(f(intervalValue))

                xyIntervalPointsPair = [xIntervalPoints, yIntervalPoints]

                intervals.append(xyIntervalPointsPair)

            else:
                xDiscretePoints.append(tsItem)
                yDiscretePoints.append(f(tsItem))

        plt.xlabel("tsValues")
        plt.ylabel("f(tsValues)")

        plt.plot(xDiscretePoints, yDiscretePoints, discreteStyle, markersize=markerSize, linewidth=lineWidth)

        for xyIntervalPointsPair in intervals:
            plt.plot(xyIntervalPointsPair[0], xyIntervalPointsPair[1], intervalStyle, markersize=markerSize, linewidth=lineWidth)

        plt.show()

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


