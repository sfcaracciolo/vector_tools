from typing import Tuple
import numpy as np 

class BiPoint:

    def __init__(self, vertices: np.ndarray | Tuple[np.ndarray]) -> None:
        self.vertices = vertices if isinstance(vertices, np.ndarray) else np.vstack(vertices)
        self._norms = np.linalg.norm(self.vertices, axis=1, keepdims=True) 
        self._normed = self.vertices / self._norms

    def get_midpoint(self) -> np.ndarray:
        return np.sum(self.vertices, axis=0)/2.

    def get_distance(self) -> np.ndarray:
        v = np.diff(self.vertices, axis=0)
        return np.linalg.norm(v)

    def get_angle(self, degree: bool=True) -> float:
        ip = np.inner(self._normed[0], self._normed[1])
        cs = np.clip(ip, -1, 1) # fix rounding issue.
        rads = np.arccos(cs)
        return np.rad2deg(rads) if degree else rads

    def get_cotangent(self) -> float:
        ip = np.inner(self.vertices[0], self.vertices[1]) 
        cp = np.cross(self.vertices[0], self.vertices[1])
        return ip/np.linalg.norm(cp)
    
    @classmethod
    def angle(cls, vertices, **kwargs):
        return cls(vertices).get_angle(**kwargs)
    
    @classmethod
    def distance(cls, vertices):
        return cls(vertices).get_distance()
    
    @classmethod
    def midpoint(cls, vertices):
        return cls(vertices).get_midpoint()
    
    @classmethod
    def cotangent(cls, vertices):
        return cls(vertices).get_cotangent()
    
class TriPoint:

    def __init__(self, vertices: np.ndarray | Tuple[np.ndarray]) -> None:
        self.vertices = vertices if isinstance(vertices, np.ndarray) else np.vstack(vertices)
        self._lengths = self.get_lenghts()
        self._vectors = self.vertices[1:] - self.vertices[0][np.newaxis, :]

    def get_area(self) -> float:
        # https://en.wikipedia.org/wiki/Heron%27s_formula
        s = np.sum(self._lengths)/2.
        mask = np.isclose(s, self._lengths) # fix rounding issue.
        aux = np.where(mask, 0., s-self._lengths)
        return np.sqrt(s*np.prod(aux))

    def get_lenghts(self) -> np.ndarray:
        vectors = np.diff(self.vertices, axis=0, append=self.vertices[0][np.newaxis,:])
        return np.linalg.norm(vectors, axis=1)

    def get_compactness(self) -> float:
        return 4*np.sqrt(3.)*self.get_area()/np.linalg.norm(self._lengths)**2

    def get_normal(self, normalize: bool = True) -> np.ndarray:
        n = np.cross(self._vectors[0], self._vectors[1])
        if normalize: n /= np.linalg.norm(n)
        return n

    def get_angle(self, **kwargs) -> float:
        return BiPoint.angle(self._vectors, **kwargs) 
    
    def get_cotangent(self) -> float:
        return BiPoint.cotangent(self._vectors)
    
    def get_is_collinear(self) -> bool:
        n = self.get_normal(normalize=False)
        return np.allclose(n, 0)
    
    def get_is_obtuse(self) -> bool:
        # https://mathworld.wolfram.com/LawofCosines.html
        # https://mathworld.wolfram.com/ObtuseTriangle.html
        a, b, c = self._lengths[0], self._lengths[1], self._lengths[2] 
        return (a**2+c**2 < b**2) or (a**2+b**2 < c**2) or (b**2+c**2 < a**2)
    
    def get_barycentric_region(self) -> float:
        return self.get_area()/3. 

    def get_voronoi_region(self) -> float:
        a, _, c = self._lengths[0], self._lengths[1], self._lengths[2] 
        p, r, q = self.vertices[0], self.vertices[1], self.vertices[2] 
        return (1/8.) * (a**2 * BiPoint.cotangent((p-q, r-q)) + c**2 * BiPoint.cotangent((p-r, q-r)))
    
    def get_mixed_voronoi_region(self) -> float:
        # http://rodolphe-vaillant.fr/entry/20/compute-harmonic-weights-on-a-triangular-mesh#mixed_voro_area
        if not self.get_is_obtuse(): return self.get_voronoi_region()
        a, b, c = self._lengths[0], self._lengths[1], self._lengths[2]
        return self.get_area()/2. if (a**2+c**2 < b**2) else self.get_area()/4.

    @classmethod
    def is_collinear(cls, vertices):
        return cls(vertices).get_is_collinear()
    
    @classmethod
    def normal(cls, vertices, **kwargs):
        return cls(vertices).get_normal(**kwargs)
    
    @classmethod
    def area(cls, vertices):
        return cls(vertices).get_area()
