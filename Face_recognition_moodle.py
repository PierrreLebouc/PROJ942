# test programm 

from pathlib import Path
from typing import Callable, Optional, Tuple

# import numpy and matplotlib colormaps
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#from skimage import io
import matplotlib.cm as cm

import cv2

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.ANTIALIAS

############

PreprocessFn = Callable[[Path, Optional[Tuple[int, int]]], Optional[np.ndarray]]


def read_images(path, sz=None, preprocess_fn: Optional[PreprocessFn] = None):
    base_path = Path(path)
    if not base_path.is_dir():
        raise ValueError(f"Chemin base images invalide: {base_path}")

    X, y, label_names = [], [], []
    for label, subject_dir in enumerate(sorted(d for d in base_path.iterdir() if d.is_dir())):
        label_names.append(subject_dir.name)
        for image_path in sorted(p for p in subject_dir.iterdir() if p.is_file()):
            prepared = None
            if preprocess_fn is not None:
                try:
                    prepared = preprocess_fn(image_path, sz)
                except Exception as exc:  # pragma: no cover - diagnostic aide
                    print(f"Prétraitement échoué pour {image_path}: {exc}")
                    prepared = None
            if prepared is not None:
                X.append(prepared)
                y.append(label)
                continue

            try:
                with Image.open(image_path) as im:
                    im = im.convert("L")
                    if sz is not None:
                        im = im.resize(sz, RESAMPLE)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(label)
            except Exception as exc:
                print(f"Impossible de charger {image_path}: {exc}")
    if not X:
        raise RuntimeError(f"Aucune image valide trouvée sous {base_path}")
    return X, y, label_names


def load_image_as_array(image_path, target_size=None):
    image_path = Path(image_path)
    with Image.open(image_path) as im:
        im = im.convert("L")
        if target_size is not None and im.size != target_size:
            im = im.resize(target_size, RESAMPLE)
        return np.asarray(im, dtype=np.uint8)
    
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
    rangeX = float(maxX - minX)
    if rangeX < 1e-12:
        X = np.full_like(X, low, dtype=np.float32)
    else:
        # normalize to [0...1].
        X = X - float(minX)
        X = X / rangeX
        # scale to [ low...high ].
        X = X * (high - low)
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


def main():
    project_root = Path(__file__).resolve().parent
    dataset_path = project_root / "Base_Visages"
    print(f"Chargement de la base de visages depuis {dataset_path} ...")

    X, y, label_names = read_images(dataset_path)
    image_height, image_width = X[0].shape
    print(f"{len(X)} images chargées pour {len(label_names)} personnes.")
    print(f"Taille des images: {image_width}x{image_height} (LxH).")

    # Example 1: afficher une image de la base
    num_image = 0
    titre = f"image {num_image} - classe {label_names[y[num_image]]}"
    cv2.imshow(titre, X[num_image])
    cv2.waitKey(0)

    # Exemple 2: calcul PCA / eigenfaces
    D, W, mu = pca(asRowMatrix(X), y)
    eigen_count = min(16, W.shape[1])
    eigenfaces = []
    for i in range(eigen_count):
        e = W[:, i].reshape(X[0].shape)
        eigenfaces.append(normalize(e, 0, 255))
    subplot(
        title=f"{eigen_count} Eigenfaces (Base_Visages)",
        images=eigenfaces,
        rows=4,
        cols=4,
        sptitle="Eigenface",
        filename="python_pca_eigenfaces.png",
    )

    # Exemple 3: reconstruction progressive
    max_components = min(W.shape[1], len(X))
    steps = [i for i in range(10, max_components, 20)]
    if not steps:
        steps = list(range(1, min(max_components, 5) + 1))
    reconstructions = []
    for numEvs in steps[:16]:
        P = project(W[:, 0:numEvs], X[num_image].reshape(1, -1), mu)
        R = reconstruct(W[:, 0:numEvs], P, mu).reshape(X[num_image].shape)
        reconstructions.append(normalize(R, 0, 255))
    subplot(
        title="Reconstruction (Base_Visages)",
        images=reconstructions,
        rows=4,
        cols=4,
        sptitle="Eigenvectors",
        sptitles=steps[: len(reconstructions)],
        colormap=cm.gray,
        filename="python_pca_reconstruction.png",
    )

    # Exemple 4: prédiction sur un petit jeu de test
    model = EigenfacesModel(X, y, num_components=min(150, len(X)))

    def describe_label(label_idx):
        return label_names[label_idx] if 0 <= label_idx < len(label_names) else str(label_idx)

    test_samples = []
    for label, subject_dir in enumerate(sorted(d for d in dataset_path.iterdir() if d.is_dir())):
        candidates = sorted(p for p in subject_dir.iterdir() if p.is_file())
        if not candidates:
            continue
        test_samples.append((candidates[0], label))
        if len(test_samples) >= 5:
            break

    external_test_dir = project_root / "Images de test du programme de reconnaissance de visages-20251013"
    if external_test_dir.is_dir():
        external_expectations = {
            "cl_14_im_0.pgm": 14,
            "cl_0_im_3_a.jpg": 0,
            "cl_0_im_3_b.jpg": 0,
            "cl_26_im_5_a.jpg": 26,
            "cl_26_im_5_b.jpg": 26,
            "cl_26_im_5_c.jpg": 26,
            "cl_44_im_0.pgm": 44,
            "cl_41_im_0.pgm": 41,
            "cl_43.pgm": 43,
        }
        for filename, expected in external_expectations.items():
            img_path = external_test_dir / filename
            if img_path.exists():
                test_samples.append((img_path, expected))

    target_size = (image_width, image_height)
    for img_path, expected_label in test_samples:
        try:
            test_image = load_image_as_array(img_path, target_size=target_size)
        except Exception as exc:
            print(f"Lecture impossible pour {img_path}: {exc}")
            continue
        prediction = model.predict(test_image)
        expected_name = describe_label(expected_label)
        predicted_name = describe_label(prediction)
        print(
            f"{img_path.name} -> attendu {expected_name} "
            f"(classe {expected_label}) / prédit {predicted_name} (classe {prediction})"
        )
        cv2.imshow(f"test {img_path.name}", test_image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
