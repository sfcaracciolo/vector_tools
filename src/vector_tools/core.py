import numpy as np 

def midpoint(vertices: np.ndarray) -> np.ndarray:
    return np.sum(vertices, axis=0)/2.

def distance(vertices: np.ndarray) -> np.ndarray:
    v = np.diff(vertices, axis=0)
    return np.linalg.norm(v)

def angle(vertices: np.ndarray, degree=True) -> float:
    pi = np.inner(vertices[0], vertices[1])
    cs = np.clip(pi, -1, 1) # fix rounding issue.
    rads = np.arccos(cs)
    return np.rad2deg(rads) if degree else rads

def triangle_area(lengths: np.ndarray) -> float:
    # https://en.wikipedia.org/wiki/Heron%27s_formula
    s = np.sum(lengths)/2.
    mask = np.isclose(s, lengths) # fix rounding issue.
    aux = np.where(mask, 0., s-lengths)
    return np.sqrt(s*np.prod(aux))

def triangle_compactness(vertices: np.ndarray) -> float:
    vectors = np.diff(vertices, axis=0, append=vertices[0][np.newaxis,:])
    lengths = np.linalg.norm(vectors, axis=1)
    area = triangle_area(lengths)
    return 4*np.sqrt(3.)*area/np.sum(lengths**2)

def triangle_normal(vertices: np.ndarray, normalize: bool = True) -> np.ndarray:
    vectors = np.diff(vertices, axis=0)
    n = np.cross(vectors[0], vectors[1])
    if normalize: n /= np.linalg.norm(n)
    return n

def are_collinear(vertices: np.ndarray) -> bool:
    n = triangle_normal(vertices, normalize=False)
    return np.allclose(n, 0)