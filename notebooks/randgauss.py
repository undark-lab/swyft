#!/usr/bin/env python

import numpy as np
import sys



def rgmodel(v,sigma=0.05):
	x = v + np.random.randn(2)*sigma
	return x

def main():
	sigma = None
	args = sys.stdin.readline()
	arg1, arg2 = args.split(' ')
	try:
		v0 = float(arg1.rstrip())
		v1 = float(arg2.rstrip())
		
	except:
		raise()

	v = np.array([v0,v1])

	if sigma is not None:
		x = rgmodel(v,sigma=sigma)
	else:
		x = rgmodel(v)

	print(str(x[0]).strip()+' '+str(x[1]).strip())



if __name__ == "__main__":
	main()


