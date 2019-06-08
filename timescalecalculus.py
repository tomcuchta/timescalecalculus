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

        # The following two dictionary data members are used for the memoization of the g_k and h_k functions of this class.
        self.memo_g_k = {}
        self.memo_h_k = {}

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
        if (k < 0):
            raise Exception("g_k(): k should never be less than 0!")

        elif (k != 0):
            currentKey = str(k) + ":" + str(t) + ":" + str(s)

            if currentKey in self.memo_g_k:
                return self.memo_g_k[currentKey]

            else:
                def g(x):
                   return self.g_k(k - 1, self.sigma(x), s)

                integralResult = self.dintegral(g, t, s)

                self.memo_g_k[currentKey] = integralResult

                return integralResult

        elif (k == 0):
            return 1

    #
    #
    # Generalized h_k polynomial from page 38 with memoization.
    #
    #
    def h_k(self, k, t, s):
        if (k < 0):
            raise Exception("h_k(): k should never be less than 0!")

        elif (k != 0):
            currentKey = str(k) + ":" + str(t) + ":" + str(s)

            if currentKey in self.memo_h_k:
                return self.memo_h_k[currentKey]

            else:
                def h(x):
                    return self.h_k(k - 1, x, s)

                integralResult = self.dintegral(h, t, s)

                self.memo_h_k[currentKey] = integralResult

                return integralResult

        elif (k == 0):
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
    # Ordinary Differential Equation solver for equations of the form
    #
    #   y'(t) = p(t)*y(t)
    #
    # where t_0 is the starting value in the timescale and
    #
    #   y_0 = y(t_0) 
    #
    # is the initial value provided by the user.
    #
    # Arguments:
    #   "y_0" is the initial value assigned to y(t_0) that is used as a starting point to evaluate the ODE.
    #
    #   "t_0" is the initial value that is considered the starting point in the timescale from which to solve subsequent points.
    #   t_0 is the value that is plugged into y to determine y_0 via: y_0 = y(t_0).
    #
    #   "t_target" is the timescale value for which y should be evaluated and returned.
    #
    #   "y_prime" is the function y'(t) of the ODE y'(t) = p(t)*y(t). 
    #   NOTE: "y_prime" MUST be defined such that the arguments ("t" and "y") appear in this order: y_prime(t, y).
    #   If this particular order is not used, then the solve_ode_for_t() function will plug in the wrong values for t and y when solving.
    #   This means that the solve_ode_for_t() function will (except in specific cases like when t = y) return an incorrect result.
    #
    # Other Variables:
    #   "t_current" is the current value of t. t must be a value in the timescale.
    #
    #   "y_current" holds the value obtained from y(t_current).
    #
    # The function will solve for the next t value until the value of y(t_target) is obtained.
    # y(t_target) is then returned.
    # Currently, t_target > t_0 is a requirement -- solving for a t_target < t_0 is not supported.
    #
    #
    def solve_ode_for_t(self, y_0, t_0, t_target, y_prime): # Note: y(t_0) = y_0
        # print("solve_ode_for_t arguments:")
        # print("y_0 =", y_0)
        # print("t_0 =", t_0)
        # print("t_target =", t_target)
        # print("")
        
        # The following is more validation code -- this is very similar to the validation code in the dIntegral function.
        #----------------------------------------------------------------------------#
        
        t_in_ts = False
        t_0_in_ts = False
        discretePoint = False
        
        for x in self.ts:
            if not isinstance(x, list) and t_target == x:
                t_in_ts = True
                
            if not isinstance(x, list) and t_0 == x: 
                discretePoint = True
                t_0_in_ts = True
            
            if isinstance(x, list) and t_target <= x[1] and t_target >= x[0]:
                t_in_ts = True
            
            if isinstance(x, list) and t_0 < x[1] and t_0 >= x[0]:
                discretePoint = False
                t_0_in_ts = True
                
            if isinstance(x, list) and t_0 == x[1]:
                discretePoint = True
                t_0_in_ts = True
            
            if t_in_ts and t_0_in_ts:                
                break
        
        if t_in_ts and not t_0_in_ts:
            raise Exception("solve_ode_for_t: t_0 is not a value in the timescale.")
        
        if not t_in_ts and t_0_in_ts:
            raise Exception("solve_ode_for_t: t_target is not a value in the timescale.")
        
        if not t_in_ts and not t_0_in_ts:
            raise Exception("solve_ode_for_t: t_0 and t_target are not values in the timescale.")
        
        if t_0 == t_target:
            return y_0
        
        elif t_0 > t_target:
            raise Exception("solve_ode_for_t: t_0 cannot be greater than t_target.")
        
        #----------------------------------------------------------------------------#
        
        t_current = t_0
        y_current = y_0
        
        ODE = integrate.ode(y_prime)
        
        while self.isInTimescale(t_current): # Technically safer than "while True:"
            if discretePoint:                
                # print("Solving right scattered point where:")
                # print("t_current =", t_current)
                # print("y_current =", y_current)
                # print("t_target =", t_target)
                # print("y_prime(t_current, y_current) =", y_prime(t_current, y_current))
                # print("self.mu(t_current) =", self.mu(t_current))
                # print()            
  
                y_sigma_of_t_current = y_current + y_prime(t_current, y_current) * self.mu(t_current)
                
                t_next = self.sigma(t_current)
                
                # print("t_next = self.sigma(t_current) =", t_next)
                # print()                
                # print("Result:")
                # print("y_sigma_of_t_current =", y_sigma_of_t_current)
                # print()
                                
                if t_target == t_next:
                    # print("t_target == t_next -> returning y_sigma_of_t_current\n")
                    return y_sigma_of_t_current
                
                if self.isDiscretePoint(t_next):
                    discretePoint = True
                    # print("[NEXT IS DISCRETE POINT]")
                    
                else:
                    # print("[NEXT IS NOT DISCRETE POINT]")
                    discretePoint = False
                
                t_current = t_next
                y_current = y_sigma_of_t_current                
                                
            else:
                # print("Solving right dense point where:")                    
                # print("t_current =", t_current)
                # print("y_current =", y_current)
                # print("t_target =", t_target)
                # print()
                                      
                ODE.set_initial_value(y_current, t_current)
                
                if self.isDiscretePoint(t_current):
                    raise Exception("t_current is NOT in a list/interval! Something went wrong!")
                
                else:
                    interval_of_t_current = self.getCorrespondingInterval(t_current)
                    
                    # print("Integration conditions:")
                    # print("t_current =", t_current)
                    # print("interval_of_t_current =", interval_of_t_current)
                    
                    if t_target <= interval_of_t_current[1] and t_target >= interval_of_t_current[0]:
                        # print("Integrating to t =", t_target)
                        # print()
                        ODE_integration_result = ODE.integrate(t_target)
                        
                        # print("Result:")
                        # print("ODE_integration_result =", ODE_integration_result)
                        # print()
                        
                        return ODE_integration_result
                    
                    elif t_target > interval_of_t_current[1]:
                        # print("Integrating to t =", interval_of_t_current[1])
                        # print()
                        ODE_integration_result = ODE.integrate(interval_of_t_current[1])
                        
                        # print("Result:")
                        # print("ODE_integration_result =", ODE_integration_result)
                        # print()
                        
                        t_current = interval_of_t_current[1]
                        y_current = ODE_integration_result
                        
                        # print("[NEXT IS DISCRETE POINT]")
                        discretePoint = True

                if not ODE.successful():
                    raise Exception("ODE.successful() returned False!");
    
    #
    #
    # This function is another version of the solve_ode_for_t() function.
    # It uses scipy.integrate.odeint to integrate over intervals rather than the scipy.integrate.ode method used by the solve_ode_for_t() function.
    # In general, it seems to be less accurate than solve_ode_for_t().
    # The additional stepSize argument (default value = 0.0001) can be used to somewhat mitigate this inaccuracy. 
    # However, even with extremely small step sizes (like stepSize = 0.0000001), solve_ode_for_t() seems to be better.
    #
    # NOTE: The scipy.integrate.odeint function requires that the argument function, y_prime(), has its arguments in a particular order.
    # The required order is exactly inverse to what is required by the scipy.integrate.ode function -- this has a high potential for user error.
    # y_prime for this function must be of the form: y_prime(y, t).
    # If y_prime(t, y) is provided, nonsensical results will be returned since the wrong values will be plugged into y and t.
    #
    #
    def solve_ode_for_t_with_odeint(self, y_0, t_0, t_target, y_prime, stepSize = 0.0001): # Note: y(t_0) = y_0
        # print("solve_ode_for_t arguments:")
        # print("y_0 =", y_0)
        # print("t_0 =", t_0)
        # print("t_target =", t_target)
        # print("")
        
        # The following is more validation code -- this is very similar to the validation code in the dIntegral function.
        #----------------------------------------------------------------------------#
        
        t_in_ts = False
        t_0_in_ts = False
        discretePoint = False
        
        for x in self.ts:
            if not isinstance(x, list) and t_target == x:
                t_in_ts = True
                
            if not isinstance(x, list) and t_0 == x: 
                discretePoint = True
                t_0_in_ts = True
            
            if isinstance(x, list) and t_target <= x[1] and t_target >= x[0]:
                t_in_ts = True
            
            if isinstance(x, list) and t_0 < x[1] and t_0 >= x[0]:
                discretePoint = False
                t_0_in_ts = True
                
            if isinstance(x, list) and t_0 == x[1]:
                discretePoint = True
                t_0_in_ts = True
            
            if t_in_ts and t_0_in_ts:                
                break
        
        if t_in_ts and not t_0_in_ts:
            raise Exception("solve_ode_for_t_with_odeint: t_0 is not a value in the timescale.")
        
        if not t_in_ts and t_0_in_ts:
            raise Exception("solve_ode_for_t_with_odeint: t_target is not a value in the timescale.")
        
        if not t_in_ts and not t_0_in_ts:
            raise Exception("solve_ode_for_t_with_odeint: t_0 and t_target are not values in the timescale.")
        
        if t_0 == t_target:
            return y_0
        
        elif t_0 > t_target:
            raise Exception("solve_ode_for_t_with_odeint: t_0 cannot be greater than t_target.")
        
        #----------------------------------------------------------------------------#
        
        t_current = t_0
        y_current = y_0
               
        while self.isInTimescale(t_current):
            if discretePoint:                
                # print("Solving right scattered point where:")
                # print("t_current =", t_current)
                # print("y_current =", y_current)
                # print("t_target =", t_target)
                # print("y_prime(y_current, t_current) =", y_prime(y_current, t_current))
                # print("self.mu(t_current) =", self.mu(t_current))
                # print()            
  
                y_sigma_of_t_current = y_current + y_prime(y_current, t_current) * self.mu(t_current)
                
                t_next = self.sigma(t_current)
                
                # print("t_next = self.sigma(t_current) =", t_next)
                # print()                
                # print("Result:")
                # print("y_sigma_of_t_current =", y_sigma_of_t_current)
                # print()
                                
                if t_target == t_next:
                    # print("t_target == t_next -> returning y_sigma_of_t_current\n")
                    return y_sigma_of_t_current
                
                if self.isDiscretePoint(t_next):
                    discretePoint = True
                    # print("[NEXT IS DISCRETE POINT]")
                    
                else:
                    # print("[NEXT IS NOT DISCRETE POINT]")
                    discretePoint = False
                
                t_current = t_next
                y_current = y_sigma_of_t_current                
                                
            else:
                # print("Solving right dense point where:")                    
                # print("t_current =", t_current)
                # print("y_current =", y_current)
                # print("t_target =", t_target)
                # print()
                
                if self.isDiscretePoint(t_current):
                    raise Exception("t_current is NOT in a list/interval! Something went wrong!")
                
                else:
                    interval_of_t_current = self.getCorrespondingInterval(t_current)
                    
                    # print("Integration conditions:")
                    # print("t_current =", t_current)
                    # print("interval_of_t_current =", interval_of_t_current)
                    
                    if t_target <= interval_of_t_current[1] and t_target >= interval_of_t_current[0]:
                        # print("Integrating to t =", t_target)
                        # print()                                             
                        
                        current_interval = np.arange(t_current, t_target + stepSize, stepSize)
                        
                        # print(current_interval)
                        # print()
                        
                        ODE_integration_result = integrate.odeint(y_prime, y_current, current_interval)
                        ODE_integration_result = ODE_integration_result[len(ODE_integration_result) - 1]
                        
                        # print("Result:")
                        # print("ODE_integration_result =", ODE_integration_result)
                        # print()
                        
                        return ODE_integration_result
                    
                    elif t_target > interval_of_t_current[1]:
                        # print("Integrating to t =", interval_of_t_current[1])
                        # print()
                        
                        current_interval = np.arange(t_current, interval_of_t_current[1] + stepSize, stepSize)
                        
                        # print(current_interval)
                        # print()
                        
                        ODE_integration_result = integrate.odeint(y_prime, y_current, current_interval)
                        ODE_integration_result = ODE_integration_result[len(ODE_integration_result) - 1]
                        
                        # print("Result:")
                        # print("ODE_integration_result =", ODE_integration_result)
                        # print()
                        
                        t_current = interval_of_t_current[1]
                        y_current = ODE_integration_result
                        
                        # print("[NEXT IS DISCRETE POINT]")
                        discretePoint = True
    #
    #
    # Ordinary Differential Equation System Solver
    #
    # Arguments:
    #   "y_0" is a list of the initial values assigned to y(t_0). These are used as a starting point from which to evaluate the system.
    #
    #   "t_0" is the initial value that is considered the starting point in the timescale from which to solve subsequent points.
    #   Initially, t_0 is the value that is plugged into y to determine y_0 via: y_0 = y(t_0).
    #
    #   "t_target" is the timescale value for which y should be evaluated and returned.
    #   Since this function solves a system of equations, the result will be a list of values that constitute the results for each of the equations in the system for t_target.
    #
    #   "y_prime" is the system of equations where each individual equation is of the form y'(t) = p(t)*y(t). 
    #   NOTE: Since this solver uses the scipy.integrate.odeint function to obtain its result, y_prime MUST be defined with a specific format.
    #   As an example, for a system of two equations, y_prime would have to defined in the following manner:
    #    
    #       def y_prime_vector(vector, t): # Argument order is required by the scipy.integrate.odeint class -- "y_prime_vector(y, vector)" will result in incorrect results
    #           x, y = vector # Extract and store the first item from "vector" into x and the second item into y
    #    
    #           dt_vector = [x*t, y*t*t] # Define the system of equations
    #
    #           return dt_vector # Return the system
    #   
    # NOTE: If the number of items in y_0 is not the same as the number of equations in y_prime, then this solver will fail.
    #
    #
    def solve_ode_system_for_t(self, y_0, t_0, t_target, y_prime, stepSize = 0.0001): # Note: y(t_0) = y_0
        # print("solve_ode_for_t arguments:")
        # print("y_0 =", y_0)
        # print("t_0 =", t_0)
        # print("t_target =", t_target)
        # print("")
        
        # The following is more validation code -- this is very similar to the validation code in the dIntegral function.
        #----------------------------------------------------------------------------#
        
        t_in_ts = False
        t_0_in_ts = False
        discretePoint = False
        
        for x in self.ts:
            if not isinstance(x, list) and t_target == x:
                t_in_ts = True
                
            if not isinstance(x, list) and t_0 == x: 
                discretePoint = True
                t_0_in_ts = True
            
            if isinstance(x, list) and t_target <= x[1] and t_target >= x[0]:
                t_in_ts = True
            
            if isinstance(x, list) and t_0 < x[1] and t_0 >= x[0]:
                discretePoint = False
                t_0_in_ts = True
                
            if isinstance(x, list) and t_0 == x[1]:
                discretePoint = True
                t_0_in_ts = True
            
            if t_in_ts and t_0_in_ts:                
                break
        
        if t_in_ts and not t_0_in_ts:
            raise Exception("solve_ode_system_for_t: t_0 is not a value in the timescale.")
        
        if not t_in_ts and t_0_in_ts:
            raise Exception("solve_ode_system_for_t: t_target is not a value in the timescale.")
        
        if not t_in_ts and not t_0_in_ts:
            raise Exception("solve_ode_system_for_t: t_0 and t_target are not values in the timescale.")
        
        if t_0 == t_target:
            return y_0
        
        elif t_0 > t_target:
            raise Exception("solve_ode_system_for_t: t_0 cannot be greater than t_target.")
        
        #----------------------------------------------------------------------------#
                
        t_current = t_0
        y_current = y_0
               
        while self.isInTimescale(t_current):
            if discretePoint:                
                # print("Solving right scattered point where:")
                # print("t_current =", t_current)
                # print("y_current =", y_current)
                # print("t_target =", t_target)
                # print("y_prime(y_current, t_current) =", y_prime(y_current, t_current))
                # print("self.mu(t_current) =", self.mu(t_current))
                # print()            
                                
                #------------------------------#
                
                # print("y_prime(y_current, t_current) =", y_prime(y_current, t_current), "self.mu(t_current) =", self.mu(t_current))
                
                temp1 = list(map(lambda x: x * self.mu(t_current), y_prime(y_current, t_current)))
                     
                # print("y_current:", y_current, "temp1:", temp1)
                
                temp2 = list(map(lambda x, y: x + y, y_current, temp1))
                
                # print("temp2 =", temp2)
                
                y_sigma_of_t_current = temp2                
                
                #------------------------------#
                    
                t_next = self.sigma(t_current)
                
                # print("t_next = self.sigma(t_current) =", t_next)
                # print()                
                # print("Result:")
                # print("y_sigma_of_t_current =", y_sigma_of_t_current)
                # print()
                                
                if t_target == t_next:
                    # print("t_target == t_next -> returning y_sigma_of_t_current\n")
                    return y_sigma_of_t_current
                
                if self.isDiscretePoint(t_next):
                    discretePoint = True
                    # print("[NEXT IS DISCRETE POINT]")
                    
                else:
                    # print("[NEXT IS NOT DISCRETE POINT]")
                    discretePoint = False
                
                t_current = t_next
                y_current = y_sigma_of_t_current                
                                
            else:
                # print("Solving right dense point where:")                    
                # print("t_current =", t_current)
                # print("y_current =", y_current)
                # print("t_target =", t_target)
                # print()
                
                if self.isDiscretePoint(t_current):
                    raise Exception("t_current is NOT in a list/interval! Something went wrong!")
                
                else:
                    interval_of_t_current = self.getCorrespondingInterval(t_current)
                    
                    # print("Integration conditions:")
                    # print("t_current =", t_current)
                    # print("interval_of_t_current =", interval_of_t_current)
                    
                    if t_target <= interval_of_t_current[1] and t_target >= interval_of_t_current[0]:
                        # print("Integrating to t =", t_target)
                        # print()                                             
                        
                        current_interval = np.arange(t_current, t_target + stepSize, stepSize)
                        
                        # print(current_interval)
                        # print()
                        
                        ODE_integration_result = integrate.odeint(y_prime, y_current, current_interval)
                        
                        # print("Result:")
                        # print("ODE_integration_result =", ODE_integration_result)
                        # print()
                        
                        ODE_integration_result = ODE_integration_result[len(ODE_integration_result) - 1]
                        
                        return ODE_integration_result
                    
                    elif t_target > interval_of_t_current[1]:
                        # print("Integrating to t =", interval_of_t_current[1])
                        # print()
                        
                        current_interval = np.arange(t_current, interval_of_t_current[1] + stepSize, stepSize)
                        
                        # print(current_interval)
                        # print()
                        
                        ODE_integration_result = integrate.odeint(y_prime, y_current, current_interval)                        
                        
                        # print("Result:")
                        # print("ODE_integration_result =", ODE_integration_result)
                        # print()
                        
                        ODE_integration_result = ODE_integration_result[len(ODE_integration_result) - 1]
                        
                        t_current = interval_of_t_current[1]
                        y_current = ODE_integration_result
                        
                        # print("[NEXT IS DISCRETE POINT]")
                        discretePoint = True
    
    #
    #
    # Utility function to avoid repeated code.
    # Simply checks whether the argument, t, is a value in the timescale.
    # If t is in the timescale, it will return True. Otherwise, it will return False.
    #
    def isInTimescale(self, t):
        for ts_item in self.ts:
            if not isinstance(ts_item, list) and ts_item == t:
                return True

            elif isinstance(ts_item, list) and  (t >= ts_item[0] and t <= ts_item[1]):
                return True
        
        return False
                
    #
    #
    # Utility function to avoid repeated code.
    # Simply checks whether the argument, t, is a discrete point or in an interval of the timescale.
    #
    #
    def isDiscretePoint(self, t):
        for x in self.ts:
            if not isinstance(x, list) and t == x:
                return True           
            
            if isinstance(x, list) and t <= x[1] and t >= x[0]:
                return False
        
        raise Exception("isDiscretePoint(): t was neither a discrete point nor in an interval!")
    
    #
    #
    # Utility function to avoid repeated code.
    # Returns the interval in which t is located for the current timescale.
    # Will raise an exception if t is not in an interval.
    #
    #
    def getCorrespondingInterval(self, t):
        for x in self.ts:
            if isinstance(x, list) and t <= x[1] and t >= x[0]:
                return x
        
        raise Exception("getCorrespondingInterval(): t not in an interval!") 
    
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
    #  These character combinations can be in any order UNLESS the reordering changes the interpretation of the character combination.
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


