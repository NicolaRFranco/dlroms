# Written by: Nicola Rares Franco, Ph.D. (MOX, Department of Mathematics, Politecnico di Milano)
# 
# Scientific articles based on this Python package:
# [1] Franco et al. (2023). A deep learning approach to reduced order modelling of parameter dependent partial differential equations
#     DOI: https://doi.org/10.1090/mcom/3781
# [2] Franco et al. (2023). Approximation bounds for convolutional neural networks in operator learning, Neural Networks.
#     DOI: https://doi.org/10.1016/j.neunet.2023.01.029
# [3] Franco et al. (2023). Mesh-Informed Neural Networks for Operator Learning in Finite Element Spaces, Journal of Scientific Computing.
#     DOI: https://doi.org/10.1007/s10915-023-02331-1
#
# Please cite the Author if you use this code for your work/research.
#
# Note: this is an auxiliary library, only to be used if mshr is not available.

import dolfin
import numpy
import gmsh
import os

class Domain(object):
    def __init__(self, main, other, operation = None):
        """Combines two domains via the specified operation."""
        self.a, self.b, self.op = main, other, operation
        self.index = 0
        self.dim = max(main.dim, other.dim)
  
    def script(self, index = 1):
        """Writes a gmsh script describing the domain."""
        res, j = self.a.script(index)
        res0, j0 = self.b.script(j)
        self.index = j0
        if(self.op == "u"):
            res0 += "BooleanUnion{%s{%d};}{%s{%d};}\n" % (self.a.entity(), self.a.index, self.b.entity(), self.b.index)
        elif(self.op == "i"):
            res0 += "BooleanIntersection{%s{%d};}{%s{%d};}\n" % (self.a.entity(), self.a.index, self.b.entity(), self.b.index)
        elif(self.op == "d"):
            res0 += "BooleanDifference{%s{%d};}{%s{%d};}\n" % (self.a.entity(), self.a.index, self.b.entity(), self.b.index)
        return res+res0, j0+1

    def __add__(self, other):
        return Domain(self, other, "u")

    def __sub__(self, other):
        return Domain(self, other, "d")

    def __mul__(self, other):
        return Domain(self, other, "i")

    def entity(self):
        if self.dim==2:
            return "Surface"
        elif self.dim==3:
            return "Volume"

class Rectangle(Domain):
    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1
        self.index = 0
        self.dim = 2

    def script(self, index = 1):
        self.index = index
        return 'Rectangle(%d) = {%f, %f, 0.0, %f, %f};\n' % (index,self.p0[0],self.p0[1],self.p1[0]-self.p0[0],
                                                             self.p1[1]-self.p0[1]), index+1

class Box(Domain):
    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1
        self.index = 0
        self.dim = 3

    def script(self, index = 1):
        self.index = index
        return 'Box(%d) = {%f, %f, %f, %f, %f, %f};\n' % (index,self.p0[0],self.p0[1],self.p0[2],
                                                          self.p1[0]-self.p0[0],self.p1[1]-self.p0[1],self.p1[2]-self.p0[2]), index+1

class Circle(Domain):
    def __init__(self, p, r = 1):
        self.p = p
        self.r = r
        self.index = 0
        self.dim = 2

    def script(self, index = 1):
        self.index = index
        return 'Disk(%d) = {%f, %f, 0.0, %f};\n' % (index,self.p[0], self.p[1], self.r), index+1

class Polygon(Domain):
    def __init__(self, *points):
        self.p = points
        if(numpy.linalg.norm(numpy.array(points[0])- numpy.array(points[-1]))>1e-15):
            raise RuntimeError("First and last point should coincide.")
        self.index = 0
        self.dim = 2

    def script(self, index = 1):
        res = ""
        self.index = index
        n = len(self.p)-1
        for p in self.p[:-1]:
            res += "Point(%d) = {%f, %f, 0.0};\n" % (self.index,p[0],p[1])
            self.index += 1
        base = self.index
        for i in range(n-1):
            res += "Line(%d) = {%d, %d};\n" % (self.index,base-n+i,base-n+1+i)
            self.index += 1
        res += "Line(%d) = {%d, %d};\n" % (self.index,base-1,base-n)
        self.index += 1
        res += "Line Loop(%d) = {" % self.index
        for i in range(n):
            res += "%d, " % (self.index-n+i)
        res = res[:-2] + "};\n"
        self.index += 1
        res += "Plane Surface(%d) = {%d};\n" % (self.index, self.index-1)
        return res, self.index+1


def mesh(domain, stepsize, structured = False):
    if(structured and domain.dim!=2):
        raise RuntimeError("Structured meshes are only available for 2D geometries.")
    code = 'SetFactory("OpenCASCADE");\nMesh.CharacteristicLengthMin = %f;\nMesh.CharacteristicLengthMax = %f;\n' % (stepsize, stepsize)
    code += domain.script()[0]
    extra = "\nTransfinite %s {%d};" %  (domain.entity(), domain.index) if structured else ""
    code += '\nPhysical %s(%d) = {%d};%s\nMesh.MshFileVersion = 2.0;' % (domain.entity(), domain.index+1, domain.index, extra)

    idf = numpy.random.randint(100000)
    print(code, file = open('%d.geo' % idf, 'w'))
    os.system("gmsh -%d %d.geo" % (domain.dim, idf))
    clear_output(wait = True)
    os.system("dolfin-convert %d.msh %d.xml" % (idf, idf))
    clear_output(wait = True)
    mesh = dolfin.cpp.mesh.Mesh("%d.xml" % idf)
    os.remove("%d.msh" % idf)
    os.remove("%d.xml" % idf)
    try:
        os.remove("%d_physical_region.xml" % idf)
    except:
        None
    os.remove("%d.geo" % id)
    return mesh
