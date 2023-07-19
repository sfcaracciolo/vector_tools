import numpy as np 
from src.vector_tools import TriPoint, BiPoint

# equilateral triangle
p0 = np.array([.5, 0, 0])
p1 = np.array([-.5, 0, 0])
p2 = np.array([0, np.sqrt(3)/2., 0])

t = TriPoint((p0, p1, p2))

assert np.allclose(np.sqrt(3)/4., t.get_area())
assert np.allclose(np.pi/3., t.get_angle(degree=False))
assert np.allclose(60, t.get_angle())
assert np.allclose(60, BiPoint.angle((p0-p1, p2-p1)))
assert np.allclose(1, t.get_compactness())

# random triangle
p0 = np.random.rand(3)
p1 = np.random.rand(3)
p2 = np.random.rand(3)

t = TriPoint((p0, p1, p2))

mp01 = BiPoint.midpoint((p0, p1))
mp02 = BiPoint.midpoint((p0, p2))
bary = np.mean([p0, p1, p2], axis=0)
at1 = TriPoint.area((p0, bary, mp01))
at2 = TriPoint.area((p0, bary, mp02))
assert np.allclose(at1+at2, t.get_barycentric_region())


print('OK')