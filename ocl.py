import pyopencl as cl
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import sys

class CL():
    def __init__(self, A, r, nx):  
        self.nx = nx
        self.na = A.shape[0]
        self.clinit()
        self.program = self.loadProgram("cl.cl")
        self.loadData(A, r)

    def clinit(self):
        
        plats = cl.get_platforms()
        from pyopencl.tools import get_gl_sharing_context_properties
        import sys
        if sys.platform == "darwin":
            self.ctx = cl.Context(properties=get_gl_sharing_context_properties(),
                                 devices=[])
        else:
            self.ctx = cl.Context(properties=[
                (cl.context_properties.PLATFORM, plats[0])]
                + get_gl_sharing_context_properties(), devices=None)
        
        #self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        return self

    def loadProgram(self, filename):
        f = open(filename, 'r')
        fstr = "".join(f.readlines())
        kernel_params = {"nx": self.nx, "na": self.na}
        return cl.Program(self.ctx, fstr % kernel_params).build()  
        
    def loadData(self, A, r):       
        mf = cl.mem_flags        
        self.A = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=A)
        self.r = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=r)
        self.I = cl.Buffer(self.ctx, mf.READ_WRITE, self.nx*4)
        self.X = cl.Buffer(self.ctx, mf.READ_WRITE, self.nx*4*4)
        self.queue.finish()
        return self

    def run(self, X):
        cl.enqueue_copy(self.queue, self.X, X)
        #cl.enqueue_acquire_gl_objects(self.queue, [self.X])
        
        self.out = np.zeros((self.nx,), dtype = np.float32)
        cl.enqueue_copy(self.queue, self.I, self.out)
        self.program.Solve(self.queue, (self.nx, self.na), None, self.A, self.r, self.X, self.I)
        #cl.enqueue_release_gl_objects(self.queue, [self.X])
        #self.queue.finish()
        cl.enqueue_barrier(self.queue)
        return self
            
    def get(self):       
        cl.enqueue_copy(self.queue, self.out, self.I)
        self.queue.finish()
        return self.out