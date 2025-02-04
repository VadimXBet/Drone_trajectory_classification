import math
import numpy as np

def turn_angle(x, y):
    T = len(x)
    angle_sum = 0
    for i in range(1, len(x)):
        if x[i]==x[i-1]:
            angle_sum += np.pi/2
            continue
        angle_sum += np.arctan((y[i]-y[i-1])/(x[i]-x[i-1]))
    return angle_sum/T

def curvature(x, y):
    t = len(x)
    k_t = 0
    for i in range(1, len(x)-1):
        a = np.linalg.norm(np.array((x[i+1],y[i+1]))-np.array((x[i-1],y[i-1])))
        b = np.linalg.norm(np.array((x[i],y[i]))-np.array((x[i-1],y[i-1])))
        c = np.linalg.norm(np.array((x[i+1],y[i+1]))-np.array((x[i],y[i])))
        arg = (a**2-b**2-c**2)/(2*b*c) if b*c >= 10**-6 else 0
        # if math.isnan(arg):
        #     arg = 0
        k_i = np.arccos(np.clip(arg, -1, 1))
        k_t += k_i
    return k_t/t

def velocity(x, y):
    t = len(x)
    dist = 0
    for i in range(1, len(x)):
        dist += np.linalg.norm(np.array((x[i],y[i]))-np.array((x[i-1],y[i-1])))
    return dist/t

def acceleration(x, y):
    t = len(x)
    dist = 0
    for i in range(1, len(x)-1):
        first = np.linalg.norm(np.array((x[i],y[i]))-np.array((x[i-1],y[i-1])))
        second = np.linalg.norm(np.array((x[i+1],y[i+1]))-np.array((x[i],y[i])))
        dist += second-first
    return dist/t

def CDF(x, y):
    t = len(x)
    cdf_sum = 0
    xc, yc = np.mean(x), np.mean(y)
    for i in range(len(x)):
        cdf_i = math.sqrt((x[i]-xc)**2 + (y[i]-yc)**2)
        cdf_sum += cdf_i
    return cdf_sum/t

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]