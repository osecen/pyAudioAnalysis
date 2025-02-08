import sys, os, numpy

def isfloat(x):
	"""
	Check if argument is float
	"""
	try:
		a = float(x)
	except ValueError:
		return False
	else:
		return True

def isint(x):
	"""
	Check if argument is int
	"""
	try:
		a = float(x)
		b = int(a)
	except ValueError:
		return False
	else:
		return a == b

def isNum(x):
	"""
	Check if string argument is numerical
	"""
	return isfloat(x) or isint(x)


def peakdet(v, delta, x = None):
    """
    Detect peaks in data based on their amplitude and other features.
    """
    
    maxtab = []
    mintab = []
    
    if x is None:
        x = numpy.arange(len(v))
    
    v = numpy.asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not numpy.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = numpy.inf, -numpy.inf
    mnpos, mxpos = numpy.nan, numpy.nan
    
    lookformax = True
    
    for i in numpy.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True
    
    return numpy.array(maxtab), numpy.array(mintab)

