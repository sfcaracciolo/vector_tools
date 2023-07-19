# Vector Tools

A project to compute algebraic operations related to segments (BiPoint class) and triangles (TriPoint class)

#### BiPoint

This class take as input two tridimensional vertices/points to allow us compute this values:

* **get_midpoint()**
* **get_distance()**
* **get_angle()**
* **get_cotangent()**

#### TriPoint

On the other hand, TriPoint class has several methods related to triangle operations, such as:

* **get_area()**
* **get_lenghts()**
* **get_compactness()** *compactness range from 0 to 1, being 1 if the triangle is equilateral and 0 if colinear.*
* **get_normal()**
* **get_is_collinear()** 
* **get_is_obtuse()**
* **get_barycentric_region()** *is the area between the barycenter, p0 and the midpoints of the incident edges to p0*
* **get_voronoi_region()** *is the area between the circumcenter, p0 and the midpoints of the incident edges to p0*
* **get_mixed_voronoi_region()** *same as voronoi region but check if triangle is obtuse*

The BiPoint methods are implemented too, considering vertices p1-p0 and p2-p0.