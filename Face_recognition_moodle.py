import errno
import sys
import os
# import numpy and matplotlib colormaps
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#from skimage import io
import matplotlib.cm as cm

import cv2

############

def read_images (path , sz= None ):
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        dirnames.sort()
        for subdirname in dirnames :
            subject_path = os.path.join(dirname, subdirname )
            for filename in os.listdir(subject_path ):
                try :
                    im = Image.open(os.path.join (subject_path , filename ))
                    im = im.convert ("L")
                    # resize to given size (if given )
                    if (sz is not None ):
                        im = im.resize(sz , Image.ANTIALIAS )
                    X.append (np.asarray (im , dtype =np.uint8 ))
                    y.append (c)
                except IOError :
                    print("I/O error ({0}) : {1} ".format(errno , os.strerror ))
                except :
                    print(" Unexpected error :", sys.exc_info() [0])
                    raise
            c = c+1
    return [X,y]
    
def asRowMatrix (X):
    if len (X) == 0:
        return np.array([])
    mat = np.empty((0 , X [0].size), dtype=X [0].dtype )
    for row in X:
        mat = np.vstack((mat,np.asarray(row).reshape(1,-1)))
    return mat
    
def asColumnMatrix (X):
    if len (X) == 0:
        return np.array ([])
    mat = np.empty ((X [0].size , 0) , dtype =X [0].dtype )
    for col in X:
        mat = np.hstack (( mat , np.asarray ( col ).reshape( -1 ,1)))
    return mat  
    
def pca(X, y, num_components =0):
    [n,d] = X.shape
    if ( num_components <= 0) or ( num_components > n):
        num_components = n
    mu = X.mean ( axis =0)
    X = X - mu
    if n>d:
        C = np.dot (X.T,X)
        [ eigenvalues , eigenvectors ] = np.linalg.eigh (C)
    else :
        C = np.dot (X,X.T)
        [ eigenvalues , eigenvectors ] = np.linalg.eigh (C)
        eigenvectors = np.dot (X.T, eigenvectors )
        for i in range (n):
            eigenvectors [:,i] = eigenvectors [:,i]/ np.linalg.norm ( eigenvectors [:,i])
    # or simply perform an economy size decomposition
    # eigenvectors , eigenvalues , variance = np.linalg.svd (X.T, full_matrices = False )
    # sort eigenvectors descending by their eigenvalue
    idx = np.argsort (- eigenvalues )
    eigenvalues = eigenvalues [idx ]
    eigenvectors = eigenvectors [:, idx ]
    # select only num_components
    eigenvalues = eigenvalues [0: num_components ].copy ()
    eigenvectors = eigenvectors [: ,0: num_components ].copy ()
    return [ eigenvalues , eigenvectors , mu]
    
def project (W, X, mu= None ):
    if mu is None :
        return np.dot (X,W)
    return np.dot (X - mu , W)
    
def reconstruct (W, Y, mu= None ):
    if mu is None :
        return np.dot(Y,W.T)
    return np.dot (Y,W.T) + mu
        
def normalize (X, low , high , dtype = None ):
    X = np.asarray (X)
    minX , maxX = np.min (X), np.max (X)
    # normalize to [0...1].
    X = X - float ( minX )
    X = X / float (( maxX - minX ))
    # scale to [ low...high ].
    X = X * (high - low )
    X = X + low
    if dtype is None :
        return np.asarray (X)
    return np.asarray (X, dtype = dtype )
    
def create_font ( fontname ='Tahoma', fontsize =10) :
    return { 'fontname': fontname , 'fontsize': fontsize }
    
def subplot (title , images , rows , cols , sptitle =" subplot ", sptitles =[] , colormap =cm.gray , ticks_visible =True , filename = None ):
    fig = plt.figure()
    # main title
    fig.text (.5 ,.95 , title , horizontalalignment ='center')
    for i in range (len( images )):
        ax0 = fig.add_subplot (rows ,cols ,(i +1) )
        plt.setp ( ax0.get_xticklabels () , visible = False )
        plt.setp ( ax0.get_yticklabels () , visible = False )
        if len ( sptitles ) == len ( images ):
            plt.title ("%s #%s" % ( sptitle , str ( sptitles[i])), create_font ('Tahoma',10))
        else :
            plt.title ("%s #%d" % ( sptitle , (i +1)), create_font ('Tahoma',10))
        plt.imshow (np.asarray ( images [i]) , cmap = colormap )
    if filename is None :
        plt.show()
    else :
        fig.savefig( filename )
        
class AbstractDistance ( object ):
    
    def __init__(self , name ):
            self._name = name
    def __call__(self ,p,q):
        raise NotImplementedError (" Every AbstractDistance must implement the __call__method.")
    @property
    def name ( self ):
        return self._name
    def __repr__( self ):
        return self._name
        
class EuclideanDistance ( AbstractDistance ): 
    def __init__( self ):
        AbstractDistance.__init__(self ," EuclideanDistance ")
    def __call__(self , p, q):
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        return np.sqrt(np.sum (np.power((p-q) ,2)))
    
class CosineDistance ( AbstractDistance ):
    def __init__( self ):
        AbstractDistance.__init__(self ," CosineDistance ")
    def __call__(self , p, q):
        p = np.asarray (p).flatten ()
        q = np.asarray (q).flatten ()
        return -np.dot(p.T,q) / (np.sqrt (np.dot(p,p.T)*np.dot(q,q.T)))
  

class BaseModel ( object ):
    def __init__ (self , X=None , y=None , dist_metric = EuclideanDistance () , num_components=0) :
        self.dist_metric = dist_metric
        self.num_components = 0
        self.projections = []
        self.W = []
        self.mu = []
        if (X is not None ) and (y is not None ):
            self.compute (X,y)
            
    def compute (self , X, y):
        raise NotImplementedError (" Every BaseModel must implement the compute method.")
        
    def predict (self , X):
        minDist = np.finfo('float').max
        minClass = -1
        Q = project ( self.W, X.reshape (1 , -1) , self.mu)
        for i in range (len( self.projections )):
            dist = self.dist_metric ( self.projections [i], Q)
            if dist < minDist :
                minDist = dist
                minClass = self.y[i]
        return minClass
        
class EigenfacesModel ( BaseModel ):
    def __init__ (self , X=None , y=None , dist_metric = EuclideanDistance () , num_components=0) :
        super ( EigenfacesModel , self ).__init__ (X=X,y=y, dist_metric = dist_metric , num_components = num_components )
        
    def compute (self , X, y):
        [D, self.W, self.mu] = pca ( asRowMatrix (X),y, self.num_components )
        # store labels
        self.y = y
        # store projections
        for xi in X:
            self.projections.append ( project ( self.W, xi.reshape (1 , -1) , self.mu))            

##########################
##########################
# main
        
# read all images within the database 
        
# append tinyfacerec to module search path
sys.path.append ("..")

# !! PATH to change
# don't forget \\
# X = image list
# y = image labels
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("ATTENTION, pensez bien à changer le chemin d'accès à la base d'images !!")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
[X,y] = read_images ("E:\\_Proj942\\2019\\Base_visages")

##############
# example 1 showing image 'n'
num_image = 0
titre = "image "+ str(num_image)
cv2.imshow(titre,X[num_image])
cv2.waitKey(0)


##############
# perform a full pca
# D = eigenvalues , W = eigenvectors , mu = mean
[D, W, mu] = pca ( asRowMatrix(X), y)

##############
# Exemple 2 showing 16 eigenvectors from first_eigenvector to last_eigenvector
# ( note : eigenvectors are stored by column, so reshape is necessary ! )
E = []
first_eigenvector = 0
nb_eigenvectors = W.shape[1]
first_eigenvector = min(first_eigenvector,nb_eigenvectors)
last_eigenvector = min(first_eigenvector + 16,nb_eigenvectors)

for i in range(first_eigenvector, last_eigenvector,1) :
    e = W[:,i].reshape(X [0].shape)
    E.append( normalize (e ,0 ,255) )
# plot them and store the plot to " python_eigenfaces.png"
#subplot ( title =" Eigenfaces AT&T Facedatabase ", images = E, rows =4, cols =4, sptitle =" Eigenface", colormap =cm.jet , filename ="python_pca_eigenfaces.png")
subplot ( title =str(16)+" Eigenfaces (AT&T Facedatabase) ", images = E, rows =4, cols =4, sptitle =" Eigenface", filename ="python_pca_eigenfaces.png")

##############
# Example 3 : reconstruction steps
#
steps =[i for i in range(10 , min (len(X), 320) , 20)]
E = []

for i in range (min(len(steps),16)):
    numEvs = steps[i]
    P = project(W[:,0: numEvs],X[num_image].reshape(1,-1),mu)
    R = reconstruct(W[:,0:numEvs],P,mu)
    #reshape and append to plots
    R = R.reshape(X[num_image].shape)
    E.append( normalize (R ,0 ,255) )
# plot them and store the plot to " python_reconstruction.png "
subplot ( title =" Reconstruction (AT&T Facedatabase) ", images =E, rows =4, cols =4, sptitle ="Eigenvectors", sptitles =steps , colormap =cm.gray , filename ="python_pca_reconstruction.png")

##############
# Example 4 : get a prediction with different images. Take care, if class = n, directory is n+1 in database 
list_ima = ["cl_14_im_0.pgm","cl_0_im_3_a.jpg","cl_0_im_3_b.jpg","cl_26_im_5_a.jpg","cl_26_im_5_b.jpg","cl_26_im_5_c.jpg"]
expected_class = [14,0,0,26,26,26]
for i in range(0, len(list_ima)):
    imtest = Image.open(list_ima[i])
    imtest = imtest.convert ("L")
    test = np.asarray (imtest , dtype =np.uint8 )
    cv2.imshow(list_ima[i],test)
    cv2.waitKey(0)
    # model computation
    model = EigenfacesModel (X , y)
    print(" expected = ", expected_class[i], " / predicted =", model.predict(test))

cv2.destroyAllWindows()

