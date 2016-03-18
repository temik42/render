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
    def __init__(self, R, I, D, parent = None):
        super(myWidget, self).__init__(parent)
        self.createLayout(R, I, D)

    def createLayout(self, R, I, D):
        self.textedit = QTextEdit()
        self.widget = Qrender(R,I,D)
        quitbutton = QPushButton("&Quit")
        quitbutton.clicked.connect(self.close)
        runbutton = QPushButton("&Run")
        runbutton.clicked.connect(self.widget.run)
        runbutton.clicked.connect(self.append)
        
        layout = QVBoxLayout()
        layout.addWidget(self.textedit)
        layout.addWidget(self.widget)
        layout.addWidget(runbutton)
        layout.addWidget(quitbutton)
        self.setLayout(layout)
        self.textedit.append("Init")
    
    def append(self):
        self.textedit.append("Running")



class Qrender(QGLWidget):
    def __init__(self, Rn, Jn, Dn, parent = None):
        super(Qrender, self).__init__(parent)
        self.Rn = Rn
        self.Jn = Jn
        self.Dn = Dn

        self.nlines = len(self.Rn)
        
        #self.nsample = int(1e5)
        self.alpha = np.float32(1)
        self.width = 512*2
        self.height = 512*2

        
    
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
        #self.clinit()
        #self.program = self.loadProgram("cl.cl")
        #self.loadData() 
        
    
    def run(self):
        self.scrnData = np.zeros((self.width,self.height), dtype = np.float32)
        
        for ii in range(0,self.nlines):
        
            Rn = self.Rn[ii]
            Jn = self.Jn[ii]
            Dn = self.Dn[ii]

            if (Rn.shape[0] > 1):
            
                dl = np.sqrt(np.sum([(Rn[:,i] - np.roll(Rn[:,i],1))**2 for i in range(0,3)],0))
                dl[0] = 0
                l = np.cumsum(dl)
                lmin = np.min(l)
                lmax = np.max(l)

                self.nsample = int(np.ceil(1e4*(lmax-lmin)))


                li = np.random.random(self.nsample)*(lmax-lmin)+lmin
                t = np.searchsorted(l,li)
                di = (li - l[t-1])/dl[t]   

                s = np.array([np.gradient(Rn[:,i])[t-1]*(1-di) + np.gradient(Rn[:,i])[t]*di for i in range(0,3)]).T
                s1 = np.array([(Rn[t,i] - Rn[t-1,i]) for i in range(0,3)]).T

                J = Jn[t-1]*(1-di)+Jn[t]*di
                R = np.array([np.random.normal(0,Dn[t-1]*(1-di)+Dn[t]*di,self.nsample) for i in range(0,3)]).T

                r = Rn[t-1,:]  
                ds2 = np.sum([s[:,i]**2 for i in range(0,3)],0)
                dr = np.array([s1[:,i]*di for i in range(0,3)]).T

                R += np.array([-(np.sum([R[:,j]*s[:,j] for j in range(0,3)],0))*s[:,i]/ds2 + r[:,i] + dr[:,i] for i in range(0,3)]).T

                X = np.zeros((self.nsample,4), dtype = np.float32)
                X[:,3] = np.ones(self.nsample, dtype = np.float32)
                C = np.ones((self.nsample,4), dtype = np.float32)

                X[:,0:3] = R
                C[:,3] = J

                self.pos_vbo.set_array(X)
                self.pos_vbo.bind()

                self.col_vbo.set_array(C)
                self.col_vbo.bind()

                self.render()
        
        
        glReadPixels(0, 0, self.width, self.height, GL_ALPHA, GL_FLOAT, self.scrnData)
        #print np.sum(self.scrnData)
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

        glDrawArrays(GL_POINTS, 0, self.nsample)

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
        
        X = np.zeros((1,4), dtype = np.float32)
        C = np.zeros((1,4), dtype = np.float32)

        
        self.pos_vbo = vbo.VBO(data=X, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
        self.pos_vbo.bind()

        self.col_vbo = vbo.VBO(data=C, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
        self.col_vbo.bind()
        return self