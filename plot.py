from elementaryfunctions import *
import matplotlib
import matplotlib.pyplot as plt

#
# as more types of time scales are included, we will have
# to manage the logic of when to use plt.scatter and plt.plot
#

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

params = {'text.latex.preamble' : [ r'\usepackage{amsfonts}']}
plt.rcParams.update(params)

def plot(f,timescale):
	plt.scatter(timescale,[f(x) for x in timescale])
