import numpy as np 
from src.vector_tools import BiPoint 

p0 = np.array([0, 1, 0])
p1 = np.array([1, 0, 0])
b = BiPoint((p0, p1))

assert np.allclose([.5,.5,0], b.get_midpoint())
assert np.allclose(np.pi/2, b.get_angle(degree=False))
assert np.allclose(90, b.get_angle(degree=True))
assert np.allclose(np.sqrt(2), b.get_distance())
assert np.allclose(0, b.get_cotangent())

print('OK')