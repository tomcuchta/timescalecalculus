import operator
from functools import reduce # Added this because in python 3.* they changed the location of the reduce() method to the functools module
from scipy import integrate
from scipy.misc import derivative
import numpy as np
import matplotlib.pyplot as plt
import symengine
import jitcdde
import mpmath

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
                
        # The following data member allows users to access the functions of the matplotlib.pyplot interface.
        # This means that a user has more control over the plotting functionality of this class.
        # For instance, the xlabel and ylabel functions of the pyplot interface can be set via this data member.
        # Then, whenever the plot() or scatter() functions of this class are called and displayed (via plt.show()), the xlabel and ylabel will display whatever the user set them to.
        # See this resource for a list of available functionality: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.html
        self.plt = plt

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

        sumOfIntegratedIntervals = sum([self.integrate_complex(f, x[0], x[1]) for x in intervals])

        return sum([sumOfIntegratedPoints, sumOfIntegratedIntervals])

    #
    #
    # Utility function to integrate potentially infinite timescale sections of points and intervals.
    #
    #
    def compute_potentially_infinite_timescale_section(self, f, ts_generator_function, ts_generator_arguments):    
        print("ts_generator_arguments =", ts_generator_arguments)
        print()
                
        # The following argument "ts_gen_arg_mpf" has "mpf" on the end because the mpmath package passes "mpf" objects to this function as part of the nsum() function's design.
        # These objects are converted to floats via the line: ts_gen_arg = float(mpmath.nstr(ts_gen_arg_mpf, n=15)).
        # This conversion enables us to integrate/solve as usual for the generated points/intervals.
        def wrapper_function(ts_gen_arg_mpf):
            ts_gen_arg = float(mpmath.nstr(ts_gen_arg_mpf, n=15))
            ts_item = ts_generator_function(ts_gen_arg)
            next_ts_item = ts_generator_function(ts_gen_arg + 1)
            self.validate_generated_timescale_value_pair(ts_item, next_ts_item)
                       
            print("wrapper_function: ts_gen_arg =", ts_gen_arg)
            print("wrapper_function: ts_generator_arguments =", ts_generator_arguments)
            print("wrapper_function: ts_item =", ts_item)
            print("wrapper_function: next_ts_item =", next_ts_item)
            
            if isinstance(ts_item, list):
                print("wrapper_function: integrating over interval")
                
                interval_result = self.integrate_complex(f, ts_item[0], ts_item[1])
                
                step_after_interval = 0                
                
                if ts_generator_arguments[1] > ts_gen_arg:                
                    if isinstance(next_ts_item, list):
                        next_ts_item = next_ts_item[0]
                        
                    step_after_interval = (next_ts_item - ts_item[1]) * f(ts_item[1])
                
                print("interval_result =", interval_result)
                print("step_after_interval =", step_after_interval)
                
                print("*****wrapper_function: RETURNING interval_result + step_after_interval =", interval_result + step_after_interval)
                print()
                
                return interval_result + step_after_interval
            
            else:
                print("wrapper_function: calculating discrete value")
                
                discrete_result = 0
                
                if ts_generator_arguments[1] > ts_gen_arg:                    
                    if isinstance(next_ts_item, list):
                        next_ts_item = next_ts_item[0]
                    
                    discrete_result = (next_ts_item - ts_item) * f(ts_item)
                
                print("*****wrapper_function: RETURNING discrete result value =", discrete_result)
                print()
                
                return discrete_result
        
        result = mpmath.nsum(wrapper_function, ts_generator_arguments)
        
        print("----RESULT----")
        print()
        
        return result

    #
    #
    # Utility function to integrate a generated timescale section of points and intervals for t. 
    # Arguments:
    #   f: The function with which to solve the timescale for a particular target value = t.
    #
    #   t_target: The timescale value for which to solve.
    #
    #   ts_generator_function: The function that, when it is fed values where, for any value n, ts_generator_arguments[0] <= n <= ts_generator_arguments[1] is True AND (n_(m+1) - n_m) = 1.
    #                          In other words: the difference between two consecutive n-values is always 1.
    #
    #   ts_generator_arguments: A list of two numbers of the form [start, end] where start is the first value fed to the ts_generator_function and end is the last value fed to that same function.
    #                           As mentioned before, the different between any two consecutive numbers between start and end is always 1 -- this is purely a method for generating the timescale.
    #                           As an example: if ts_generator_arguments = [1, 4], then ts_generator_function will be fed the following values in the listed order: 1, 2, 3, 4.
    #                           That ts_generator_function will then return, for each value fed to it, a particular timescale section. 
    #                           compute_potentially_infinite_timescale_section_for_t() will then solve over these sections.
    #
    #   signif_count: How many previously calcuated values (that are obtained from solving integrals or discrete points) to check the "significance" of -- see "signif_threshold" for more information.
    #
    #   signif_threshold: The value that a particular calculated value (from solving an integral or discrete point) must be below to be considered "insignificant".
    #                     If m=signif_count previously calculated values are all below this signif_threshold, then the compute_potentially_infinite_timescale_section_for_t() function will
    #                     return the current result sum with a warning message.
    #
    #
    def compute_potentially_infinite_timescale_section_for_t(self, f, t_target, ts_generator_function, ts_generator_arguments, signif_count = 10, signif_threshold = 0.0000001):    
        print("ts_generator_arguments =", ts_generator_arguments)
        print()
                
        # The following argument "ts_gen_arg_mpf" has "mpf" on the end because the mpmath package passes "mpf" objects to this function as part of the nsum() function's design.
        # These objects are converted to floats via the line: ts_gen_arg = float(mpmath.nstr(ts_gen_arg_mpf, n=15)).
        # This conversion enables us to integrate/solve as usual for the generated points/intervals.
        def wrapper_function(ts_gen_arg_mpf):
            ts_gen_arg = float(mpmath.nstr(ts_gen_arg_mpf, n=15))
            ts_item = ts_generator_function(ts_gen_arg)
            next_ts_item = ts_generator_function(ts_gen_arg + 1)
            self.validate_generated_timescale_value_pair(ts_item, next_ts_item)
                       
            print("wrapper_function: ts_gen_arg =", ts_gen_arg)
            print("wrapper_function: ts_generator_arguments =", ts_generator_arguments)
            print("wrapper_function: t_target =", t_target)
            print("wrapper_function: ts_item =", ts_item)
            print("wrapper_function: next_ts_item =", next_ts_item)
                       
            if isinstance(ts_item, list):
                print("wrapper_function: integrating over interval")
                
                if ts_item[0] > t_target:
                    raise Exception("ts_item[0] = " + str(ts_item[0]) + " was greater than t_target = " + str(t_target) + " -- the timescale does not contain t_target")
                
                if t_target > ts_item[1]:
                    interval_result = self.integrate_complex(f, ts_item[0], ts_item[1])
                
                else:
                    interval_result = self.integrate_complex(f, ts_item[0], t_target)

                    print("*****wrapper_function: t_target <= ts_item[1] -> RETURNING interval_result =", interval_result)
                    
                    return {"result" : interval_result, "found_t_target" : True}
                
                step_after_interval = 0                
                
                if ts_generator_arguments[1] > ts_gen_arg:                
                    if isinstance(next_ts_item, list):
                        next_ts_item = next_ts_item[0]
                        
                    step_after_interval = (next_ts_item - ts_item[1]) * f(ts_item[1])
                
                print("interval_result =", interval_result)
                print("step_after_interval =", step_after_interval)
                
                print("*****wrapper_function: RETURNING interval_result + step_after_interval =", interval_result + step_after_interval)
                print()
                
                if t_target == next_ts_item:
                    return {"result" : interval_result + step_after_interval, "found_t_target" : True}
                    
                else:
                    return {"result" : interval_result + step_after_interval, "found_t_target" : False}
            
            else:
                print("wrapper_function: calculating discrete value")
                
                if ts_item > t_target:
                    raise Exception("ts_item = " + str(ts_item) + " was greater than t_target = " + str(t_target) + " -- the timescale does not contain t_target")
                
                discrete_result = 0
                
                if ts_generator_arguments[1] > ts_gen_arg:                    
                    if isinstance(next_ts_item, list):
                        next_ts_item = next_ts_item[0]
                    
                    discrete_result = (next_ts_item - ts_item) * f(ts_item)
                
                print("*****wrapper_function: RETURNING discrete result value =", discrete_result)
                print()
                
                if t_target == ts_item:
                    return {"result" : discrete_result, "found_t_target" : True}
                    
                else:
                    return {"result" : discrete_result, "found_t_target" : False}
        
        result = self.special_sum_function(wrapper_function, ts_generator_arguments, prior_results_significance_count = signif_count, significance_limit = signif_threshold)
        
        print("----RESULT----")
        print()
        
        return result

    #
    #
    # This function is used by the compute_potentially_infinite_timescale_section_for_t() function of this class.
    # It is responsible for summing the results obtained from solving discrete points and intervals in a generated timescale.
    # It is a "special" function because, in addition to summing, it also checks -- via the insignificant_prior_results() function -- the values of the previous n=prior_results_significance_count calculated values.
    # Calculated values that are below the significance_limit are considered "insignificant" -- if all n previously
    # calculated values are "insignificant", the special_sum_function() return the current result along with a warning message.
    # The reasoning behind having a significance_limit is so that an overflow is less likely to occur -- if result values are allowed to indefinitely become smaller, then it is likely that the precision limit on the float
    # data type is reached which results in an overflow error.
    #
    #
    def special_sum_function(self, function, ts_generator_arguments, prior_results_significance_count = 10, significance_limit = 0.0000001):
        result = 0.0
        iteration = 0
        ts_generator_arg = ts_generator_arguments[0]
        iterations_limit = ts_generator_arguments[1] - ts_generator_arguments[0]

        print("iterations_limit =", iterations_limit)
        print("prior_results_significance_count =", prior_results_significance_count)
        print("significance_limit =", significance_limit)

        prior_results = []
                
        i = 0
        
        while i < prior_results_significance_count:
            prior_results.append(0.0)
            i = i + 1

        while iteration < iterations_limit:
            print()
            print("iteration =", iteration)
            print("ts_generator_arg =", ts_generator_arg)
            
            dictionary_result = function(ts_generator_arg)
            
            prior_results[iteration % prior_results_significance_count] = dictionary_result["result"]
            result = result + dictionary_result["result"]
            
            if dictionary_result["found_t_target"] is True:
                print("special_sum_function: found_t_target -> returning result")
                break
            
            if iteration >= prior_results_significance_count:
                print("special_sum_function: checking significance of the last " + str(prior_results_significance_count) + " results:")
                if self.insignificant_prior_results(prior_results, prior_results_significance_count, significance_limit) is True:
                    print("special_sum_function: ***WARNING***: Insignificant results detected: returning result before t_target was found")                    
                    break
            
            iteration = iteration + 1
            ts_generator_arg = ts_generator_arg + 1
        
        return result

    #
    #
    # Utility function for the special sum function -- it checks the last n=prior_results_significance_count prior_results to see if they are below the significance_limit.
    # If all n prior_results are below that limit, this function will return True.
    # Else it will return False.
    #
    # NOTE: The conditions for when values are considered insignificant/significant could be expanded upon.
    #
    #
    def insignificant_prior_results(self, prior_results, prior_results_significance_count, significance_limit):
        i = 0
        
        insignificant = True
        
        while i < prior_results_significance_count:
            print("prior_results[" + str(i) + "] =", prior_results[i])
            if prior_results[i] > significance_limit:
                insignificant = False
                
            i = i + 1
        
        if insignificant:
            return True
        
        else:
            return False

    #
    #
    # Utility function to check if a pair of generated timescale values are valid.
    # "Valid" in this case means that, for any ts_item, the condition (next_ts_item > ts_item) holds.
    # This should be true regardless of whether ts_item or next_ts_item are intervals or discrete points.
    #
    #
    def validate_generated_timescale_value_pair(self, ts_item, next_ts_item):
        if isinstance(ts_item, list):
            if isinstance(next_ts_item, list):
                if ts_item[1] >= next_ts_item[0]:
                    raise Exception("ts_item[1] >= next_ts_item[0] where: ts_item[1] = " + str(ts_item[1]) + " and next_ts_item[0] = " + str(next_ts_item[0]))
            
            else:
                if ts_item[1] >= next_ts_item:
                    raise Exception("ts_item[1] >= next_ts_item where: ts_item[1] = " + str(ts_item[1]) + " and next_ts_item = " + str(next_ts_item))
        
        else:
            if isinstance(next_ts_item, list):
                if ts_item >= next_ts_item[0]:
                    raise Exception("ts_item >= next_ts_item[0] where: ts_item = " + str(ts_item) + " and next_ts_item[0] = " + str(next_ts_item[0]))
            
            else:
                if ts_item >= next_ts_item:
                    raise Exception("ts_item >= next_ts_item where: ts_item = " + str(ts_item) + " and next_ts_item = " + str(next_ts_item))

    #
    #
    # Utility function to get n values of a generated timescale.
    #
    #
    def get_n_ts_gen_values(self, ts_gen_function, n):
        i = 0

        generated_timescale = []

        while i < n:
            generated_timescale.append(ts_gen_function(i))
            i = i + 1

        return generated_timescale

    #
    #
    # Utility function to print n values of a generated timescale.
    #
    #
    def print_n_ts_gen_values(self, ts_gen_function, n):
        i = 0

        generated_timescale = []

        while i < n:
            generated_timescale.append(ts_gen_function(i))
            i = i + 1

        print("Generated timescale:")
        print(generated_timescale)
        print()

    #
    #
    # Utility function to integrate potentially complex functions.
    #
    #
    def integrate_complex(self, f, s, t, **kwargs):
        def real_component(t):
            return np.real(f(t))
            
        def imaginary_component(t):
            return np.imag(f(t))
        
        real_result = float(mpmath.nstr(mpmath.quad(real_component, [s, t], **kwargs), n=15))        
        imaginary_result = float(mpmath.nstr(mpmath.quad(imaginary_component, [s, t], **kwargs), n=15))        
        
        if imaginary_result == 0:
            return real_result
        
        else:
            return real_result + 1j*imaginary_result
            
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
    # Cylinder transformation from definition 2.21
    #
    #
    def cyl(self, t, z):
        if (self.mu(t) == 0):
            return z
        
        else:
            return 1/self.mu(t) * np.log(1 + z*self.mu(t))

    #
    #
    # Delta exponential based on definition 2.30
    #
    #
    def dexp_p(self, p, t, s):        
        def f(t):
            return self.cyl(t, p(t))
               
        return np.exp(self.dintegral(f, t, s))

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

        result = ((dexp_p1 + dexp_p2) / 2)
        
        if np.imag(result) == 0:
            return np.real(result)
        
        else:
            return result

    #
    #
    # The forward-derivative sine trigonometric function.
    #
    #
    def dsin_p(self, p, t, s):
        dexp_p1 = self.dexp_p(lambda x: p(x) * 1j, t, s)

        dexp_p2 = self.dexp_p(lambda x: p(x) * -1j, t, s)

        result = ((dexp_p1 - dexp_p2) / 2j)

        if np.imag(result) == 0:
            return np.real(result)
        
        else:
            return result

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
    # Delay Differential Equation Solver (currently unfinished)
    #
    #
    def solve_dde_for_t(self, y_values, t_0, t_target, y_prime, JiTCDDE=None, stepSize=0.01):
        print("solve_dde_for_t arguments:")
        print("y_0 = y_values[t_0] =", y_values[t_0])
        print("t_0 =", t_0)  
        print("y_values =", y_values)
        print("t_target =", t_target)
        print("")
        
        # The following is validation code for the argument "y_values".
        #----------------------------------------------------------------------------#
        
        for y_value_key in y_values:
            if not self.isInTimescale(y_value_key):
                raise Exception("The initial y value, t =", y_value_key, " is not in the timescale")
        
        y_0 = y_values[t_0]
        
        #----------------------------------------------------------------------------#
        
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
            raise Exception("solve_dde_for_t: t_0 is not a value in the timescale.")
        
        if not t_in_ts and t_0_in_ts:
            raise Exception("solve_dde_for_t: t_target is not a value in the timescale.")
        
        if not t_in_ts and not t_0_in_ts:
            raise Exception("solve_dde_for_t: t_0 and t_target are not values in the timescale.")
        
        if t_0 == t_target:
            print("t_0 == t_target -> returning y_0\n")
            return y_0
        
        elif t_0 > t_target:
            raise Exception("solve_dde_for_t: t_0 cannot be greater than t_target.")
        
        #----------------------------------------------------------------------------#
        
        t_current = t_0
        # y_current = y_0
        
        all_results = []
        
        past_points = []
        
        while self.isInTimescale(t_current):
            if discretePoint:               
                print("Solving right scattered point where:")
                print("t_current =", t_current)
                print("y_current = y_values[t_current] =", y_values[t_current])
                print("t_target =", t_target)
                print("y_prime(t_current, y_values) =", y_prime(t_current, y_values))
                print("self.mu(t_current) =", self.mu(t_current))
                print()            
                
                #--------#
                
                # y_sigma_of_t_current = y_current + y_prime(t_current, y_current) * self.mu(t_current)
                
                # t_next = self.sigma(t_current)       
                
                #--------#
                
                #--------------------#
                
                y_sigma_of_t_current = y_values[t_current] + y_prime(t_current, y_values) * self.mu(t_current)
                
                t_next = self.sigma(t_current) 
                
                #--------------------#

                print("t_next = self.sigma(t_current) =", t_next)
                print()                
                print("Result:")
                print("y_sigma_of_t_current =", y_sigma_of_t_current)
                print()
                
                all_results.append(y_sigma_of_t_current)
                                
                if t_target == t_next:
                    print("t_target == t_next -> returning y_sigma_of_t_current\n")
                    # return y_sigma_of_t_current
                    return all_results
                
                if self.isDiscretePoint(t_next):
                    discretePoint = True
                    print("[NEXT IS DISCRETE POINT]")
                    print()
                    
                else:
                    print("[NEXT IS NOT DISCRETE POINT]")
                    print()                    
                    discretePoint = False
                
                past_point = [t_current, [y_values[t_current]], [y_sigma_of_t_current]]
                past_points.append(past_point)
                
                y_values[t_next] = y_sigma_of_t_current
                
                t_current = t_next
                # y_current = y_sigma_of_t_current
                
            else:
                print("Solving right dense point where:")                    
                print("t_current =", t_current)
                print("y_current = y_values[t_current] =", y_values[t_current])
                print("t_target =", t_target)
                print()
                
                if self.isDiscretePoint(t_current):
                    raise Exception("t_current is NOT in a list/interval! Something went wrong!")
                
                else:
                    interval_of_t_current = self.getCorrespondingInterval(t_current)
                    
                    print("Integration conditions:")
                    print("t_current =", t_current)
                    print("interval_of_t_current =", interval_of_t_current)
                    
                    if t_target <= interval_of_t_current[1] and t_target >= interval_of_t_current[0]:
                        print("Integrating to t =", t_target)
                        print()                                             
                        
                        current_interval = np.arange(t_current, t_target + stepSize, stepSize)
                        
                        print(current_interval)
                        print()
                        
                        DDE_integration_result = []                        
                        JiTCDDE = self.updateJiTCDDE(JiTCDDE, past_points)                      
                        past_points = []
                                                
                        for time in current_interval:
                            if time <= t_target:
                                DDE_integration_result = JiTCDDE.integrate_blindly(time)
                                all_results.append(DDE_integration_result[0])
                                print("time =", time, " |  integration_result =", DDE_integration_result)
                        
                        # if t_target != JiTCDDE.t:
                            # raise Exception("t_target != JiTCDDE.t: t_target =", t_target, "| JiTCDDE.t =", JiTCDDE.t)
                        
                        #---Testing-Code-Start---#
                        
                        t_current = t_target # The following should hold barring accuracy limitations: t_target != JiTCDDE.t
                        y_values[t_current] = DDE_integration_result[0]
                        
                        print("t_current =", t_current)
                        print("JiTCDDE.t =", JiTCDDE.t)                          
                        print("y_current = y_values[t_current] =", y_values[t_current])
                        print()
                        
                        #---Testing-Code-End---#
                        
                        print("Result:")
                        print("time =", t_current, "| DDE_integration_result =", DDE_integration_result)
                        print()
                        
                        # return DDE_integration_result[len(DDE_integration_result) - 1]
                        return all_results
                    
                    elif t_target > interval_of_t_current[1]:
                        print("Integrating to t =", interval_of_t_current[1])
                        print()
                        
                        current_interval = np.arange(t_current, interval_of_t_current[1] + stepSize, stepSize)
                        
                        print(current_interval)
                        print()
                        
                        DDE_integration_result = []                        
                        JiTCDDE = self.updateJiTCDDE(JiTCDDE, past_points)                        
                        past_points = []
                                                
                        for time in current_interval:
                            if time <= t_target:
                                DDE_integration_result = JiTCDDE.integrate_blindly(time)
                                all_results.append(DDE_integration_result[0])
                                print("time =", time, " |  integration_result =", DDE_integration_result)
                        
                        # if interval_of_t_current[1] != JiTCDDE.t:
                            # raise Exception("interval_of_t_current[1] != JiTCDDE.t: interval_of_t_current[1] =", interval_of_t_current[1], "| JiTCDDE.t =", JiTCDDE.t)
                        
                        t_current = interval_of_t_current[1] # The following should hold barring accuracy limitations: interval_of_t_current[1] == JiTCDDE.t
                        y_values[t_current] = DDE_integration_result[0]
                        
                        print("t_current =", t_current)
                        print("JiTCDDE.t =", JiTCDDE.t)                          
                        print("y_current = y_values[t_current] =", y_values[t_current])
                        print()
                        
                        print("Result:")
                        print("time =", t_current, "| DDE_integration_result =", DDE_integration_result)
                        print()
                        
                        print("[NEXT IS DISCRETE POINT]")
                        print()                        
                        discretePoint = True
    
    #
    #
    # Validation function that checks whether the value of a delay function (which is passed to this function as an argument) is in the timescale.
    # If an error is provided, then this function will use the isInTimescaleWithError() function to determine if the value is in the timescale.
    # If no error is provided, then the isInTimescale() function will be used.
    # This function is primarily intended to be used when defining a y_prime function to be used with the solve_dde_for_t() function of this class.
    #
    #    
    def delay(self, delay_value, error=None):    
        if error is None:
            if not self.isInTimescale(delay_value):
                raise Exception("The delay value =", delay_value, " was not in the timescale")
        
        else:
            if not self.isInTimescaleWithError(delay_value, error):
                raise Exception("The delay value =", delay_value, " was not in the timescale with error =", error)
                
        return delay_value
    
    #
    #
    # Utility function to avoid repeated code.
    # Sets up the "jitcdde" class to integrate over intervals.
    # For more information see: https://jitcdde.readthedocs.io/en/stable/#the-main-class
    # This function is used by the solve_dde_for_t() function.
    #
    #    
    def initializeJiTCDDE(self, y_prime_jitcdde, past_function, arg_max_delay, arg_times_of_interest, c_backend):    
        DDE = jitcdde.jitcdde(y_prime_jitcdde, max_delay=arg_max_delay)                  
        DDE.past_from_function(past_function, times_of_interest=arg_times_of_interest)

        if c_backend == False:
            DDE.generate_lambdas()  

        print()
        
        # print("state:")
        # x = DDE.get_state()
        
        # for y in x:
            # print(y)
        
        # print()

        return DDE

    #
    #
    # Utility function to avoid repeated code.
    # Updates the past points of the "jitcdde" class.
    # For more information see: https://jitcdde.readthedocs.io/en/stable/#_jitcdde.jitcdde.add_past_point
    # This function is used by the solve_dde_for_t() function.
    #
    #    
    def updateJiTCDDE(self, DDE, past_points):
        # print("state:")
        # x = DDE.get_state()
        
        # for y in x:
            # print(y)
        # print()
    
        print("past points:")
        for past_point in past_points:
            time = past_point[0]
            state = past_point[1]
            derivative = past_point[2]            
            
            print("time:", time, "| state:", state, "| derivative:", derivative)            
            
            DDE.add_past_point(time, state, derivative)
        
        print()
        
        return DDE 
    
    #
    #
    # Utility function to avoid repeated code.
    # Simply checks whether the argument, t, is close to a value in the timescale.
    # What "close" means is defined by the argument "error".
    # If t is close to a value in the timescale, it will return True. Otherwise, it will return False.
    #
    #
    def isInTimescaleWithError(self, t, error=0.000000000000001):
        for ts_item in self.ts:
            if not isinstance(ts_item, list):
                if ts_item >= (t - error) and ts_item <= (t + error):
                    return True

            elif isinstance(ts_item, list):
                if (t >= (ts_item[0] - error) and t <= (ts_item[1] + error)):
                    return True
        
        return False
    
    #
    #
    # Utility function to avoid repeated code.
    # Simply checks whether the argument, t, is a value in the timescale.
    # If t is in the timescale, it will return True. Otherwise, it will return False.
    #
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
    # Required argument:
    #   f:
    #   The function that will determine the y values of the graph - the x values are determined by the current timescale values.
    #
    # Optional arguments:
    #   stepSize:
    #   The accuracy to which the intervals are drawn in the graph - the smaller the value, the higher the accuracy and overhead.
    #  
    #   discreteStyle, intervalStyle:
    #   These arguments determine the color, marker, and line styles of the graph.
    #   They accept string arguments of 3-part character combinations that represent a color, marker style, and line style.
    #   For instance the string "-r." indicates that the current (x, y) points should be plotted with a connected line
    #   (represented by the "-"), in a red color (represented by the "r"), and with a point marker (represented by the ".").
    #   These character combinations can be in any order UNLESS the reordering changes the interpretation of the character combination.
    #   For instance, the string "r-." does not produce the same result as the string "-r.".
    #   This is because "-." indicates a "dash-dot line style" and is therefore no longer interpreted as a "point marker" with a "solid line style".
    #   See the notes section of this resource for more information: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
    #
    #   **kwargs:
    #   This argument gives the user access to all the arguments of the matplotlib.pyplot.plot function (this includes markersize, linewidth, color, label, and dashes).
    #   For a list of all available parameters, see: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
    #
    # NOTE: To display plots that are created with this function, call the show() function of the plt data member of this class.
    #
    #
    def plot(self, f, stepSize=0.01, discreteStyle='b.', intervalStyle='r-', **kwargs):
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

        plt.plot(xDiscretePoints, yDiscretePoints, discreteStyle, **kwargs)    
        
        labeled = False
        removedLabel = False
        
        if "label" in kwargs:
            if discreteStyle == intervalStyle:            
                kwargs.pop("label")
                removedLabel = True       

        for xyIntervalPointsPair in intervals:
            if "label" in kwargs:
                if labeled == True and removedLabel == False:                
                    kwargs.pop("label")    
                    removedLabel = True                  
                
            plt.plot(xyIntervalPointsPair[0], xyIntervalPointsPair[1], intervalStyle, **kwargs)
            
            labeled = True
    
    #
    #
    # Scatter plotting functionality.
    #
    # Required argument:
    #   f:
    #   The function that will determine the y values of the graph - the x values are determined by the current timescale values.
    #
    # Optional arguments:
    #   stepSize:
    #   The accuracy to which the intervals are drawn in the graph - the smaller the value, the higher the accuracy and overhead.
    #
    #   **kwargs:
    #   This argument gives the user access to all the arguments of the matplotlib.pyplot.scatter function (this includes marker, color, and label).
    #   For a list of all available parameters, see: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html
    #
    # NOTE: To display plots that are created with this function, call the show() function of the plt data member of this class.
    #
    #
    def scatter(self, f, stepSize=0.01, **kwargs):
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
            
        plt.scatter(xDiscretePoints, yDiscretePoints, **kwargs)    

        if "label" in kwargs:
            if "color" in kwargs:
                kwargs.pop("label")
                
        for xyIntervalPointsPair in intervals:            
            plt.scatter(xyIntervalPointsPair[0], xyIntervalPointsPair[1], **kwargs)
    
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


