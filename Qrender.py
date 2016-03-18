import numpy as np
import scipy.misc
from scipy.spatial import Delaunay

from PyQt4.QtOpenGL import *
from PyQt4.QtGui import *

from OpenGL.GL import *
from OpenGL.GL.ARB.framebuffer_object import *
from OpenGL.GL.EXT.framebuffer_object import *

import sys
import pyopencl as cl


class myWidget(QWidget):
    def __init__(self, R, I, parent = None):
        super(myWidget, self).__init__(parent)
        self.createLayout(R, I)

    def createLayout(self, R, I):
        self.textedit = QTextEdit()
        widget = Qrender(R,I)
        quitbutton = QPushButton("&Quit")
        quitbutton.clicked.connect(self.close)
        runbutton = QPushButton("&Run")
        runbutton.clicked.connect(widget.run)
        runbutton.clicked.connect(self.append)
        
        layout = QVBoxLayout()
        layout.addWidget(self.textedit)
        layout.addWidget(widget)
        layout.addWidget(runbutton)
        layout.addWidget(quitbutton)
        self.setLayout(layout)
        self.textedit.append("Init")
    
    def append(self):
        self.textedit.append("Running")



class Qrender(QGLWidget):
    def __init__(self, Rn, Jn, parent = None):
        super(Qrender, self).__init__(parent)
                
        tes = Delaunay(Rn).simplices
        self.na = tes.shape[0]

        R = Rn[tes,:]
        J = Jn[tes]
        
        self.R = np.array(R)
        A1 = np.linalg.inv(np.transpose([[self.R[:,i,j]-self.R[:,3,j]
                                              for j in range(0,3)] for i in range(0,3)],[2,1,0]))
        
        #self.na = A1.shape[0]
        self.A = np.zeros((self.na,4,4), dtype = np.float32)
        self.A[:,0:3,0:3] = A1
        self.A[:,:,3] = J
        self.A[:,3,0:3] = R[:,3,:]

        
        #print self.na
        
        self.nsample = 1000
        self.alpha = np.float32(1)
        self.width = 512
        self.height = 512

        
    
    def initializeGL(self):        
        rendertarget=glGenTextures(1)

        glBindTexture( GL_TEXTURE_2D, rendertarget );
        glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                         GL_REPEAT);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,
                         GL_REPEAT );
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        
        glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA16F,self.width,self.height,0,GL_RGBA,
                 GL_FLOAT, None)
        
        glGenFramebuffers(1)
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 1)
        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, 
                              GL_TEXTURE_2D, rendertarget, 0)
        
        #glPushAttrib(GL_VIEWPORT_BIT)    
        
        glViewport(0, 0, self.width, self.height)
        glOrtho(0,1,0,1,-1,0)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        
        self.loadVBO()
        self.clinit()
        self.program = self.loadProgram("cl.cl")
        self.loadData() 
        
    
    
    def run(self):
        ni = 100
        self.scrnData = np.zeros((self.width,self.height), dtype = np.float32)
        
        for ii in range(0,ni):
            for jj in range(0,ni):
                r = np.random.random([self.nsample,3])
                r[:,0]=(r[:,0]+ii)/ni
                r[:,1]=(r[:,1]+jj)/ni
                
                self.X = np.zeros((self.nsample,4), dtype = np.float32)
                self.X[:,0:3] = r
                self.X[:,3] = 1.
                
                self.I = np.zeros((self.nsample,4), dtype = np.float32)
                self.I[:,0:3] = 1.
                #self.I[:,3] = 0.
                                
                cl.enqueue_acquire_gl_objects(self.queue, [self.X_cl,self.I_cl])
                cl.enqueue_copy(self.queue, self.X_cl, self.X)
                cl.enqueue_copy(self.queue, self.I_cl, self.I)
                self.program.Solve(self.queue, (self.nsample, self.na), None, self.A_cl, self.X_cl, self.I_cl, self.alpha)
                cl.enqueue_release_gl_objects(self.queue, [self.X_cl,self.I_cl])
                self.queue.finish()             
                
                self.render()
            glReadPixels(0, 0, self.width, self.height, GL_ALPHA, GL_FLOAT, self.scrnData)
            scipy.misc.imsave('render.png', np.flipud(self.scrnData))
          
        


        
        
    
    def clinit(self):
        plats = cl.get_platforms()
        from pyopencl.tools import get_gl_sharing_context_properties
        #import sys
        if sys.platform == "darwin":
            self.ctx = cl.Context(properties=get_gl_sharing_context_properties(),
                                 devices=[])
        else:
            self.ctx = cl.Context(properties=[
                (cl.context_properties.PLATFORM, plats[0])]
                + get_gl_sharing_context_properties(), devices=None)
        
        self.queue = cl.CommandQueue(self.ctx)
        return self
    
    
    def loadProgram(self, filename):
        f = open(filename, 'r')
        fstr = "".join(f.readlines())
        kernel_params = {"nx": self.nsample, "na": self.na}
        return cl.Program(self.ctx, fstr % kernel_params).build()  
    

    
    def render(self):
        self.col_vbo.bind()
        glColorPointer(4, GL_FLOAT, 0, None)
        self.pos_vbo.bind()
        glVertexPointer(4, GL_FLOAT, 0, None)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        glDrawArrays(GL_POINTS, 0, self.X.shape[0])

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

    
    def loadData(self):       
        mf = cl.mem_flags        
        self.A_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.A)
        
        self.pos_vbo.bind()
        self.X_cl = cl.GLBuffer(self.ctx, mf.READ_WRITE, self.pos_vbo.buffer)
        
        self.col_vbo.bind()
        self.I_cl = cl.GLBuffer(self.ctx, mf.READ_WRITE, self.col_vbo.buffer)
        self.queue.finish()
        return self
    
    
    def loadVBO(self):    
        from OpenGL.arrays import vbo
        
        self.X = np.zeros((self.nsample,4), dtype = np.float32)
        self.color = np.ones((self.nsample,4), dtype = np.float32)
        
        self.pos_vbo = vbo.VBO(data=self.X, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
        self.pos_vbo.bind()

        self.col_vbo = vbo.VBO(data=self.color, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
        self.col_vbo.bind()
        return self