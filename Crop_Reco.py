import numpy as np
import cv2
import requests
from matplotlib import pyplot as plt
import os
from pathlib import Path
from typing import Callable, Optional, Tuple
from PIL import Image

base_path = "/Users/pedron/Desktop/Polytech/PROJ942/Base/IMG_1.JPG"
output_dir = "/Users/pedron/Desktop/Polytech/PROJ942/Images_de_test"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "1.pgm")

data_root = Path("/Users/pedron/Desktop/Polytech/PROJ942")
dataset_path = data_root / "Base_Visages"
test_path = data_root / "Images_de_test"


def read_image(path):
    img = cv2.imread(path)
    height, width, depth = img.shape
    return img






# -----------------------------------------------------------------------------------
#                                    Crop image
# -----------------------------------------------------------------------------------

def crop_image(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image.shape

    resize_1 = cv2.resize(gray_image, (600, 800)) 
    

    #load the pre-trained classifier
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    #perform the classifier
    face = face_classifier.detectMultiScale(
        resize_1, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    x1 = face[0][0] - round(face[0][2] / 48)
    y1 = face[0][1] - round(face[0][2] / 6)
    x2 = face[0][0] + face[0][2] + round(face[0][2] / 48)
    y2 = face[0][1] + face[0][3] + round(face[0][2] / 6)

    cropped_image = resize_1[y1:y2, x1:x2]

    final = cv2.resize(cropped_image, (92, 112)) 
    print("Image redimensionnée à :", final.shape[:2])
    
    return final

def crop_image_creation(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image.shape

    resize_1 = cv2.resize(gray_image, (600, 800)) 

    #load the pre-trained classifier
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    #perform the classifier
    face = face_classifier.detectMultiScale(
        resize_1, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    x1 = face[0][0] - round(face[0][2] / 48)
    y1 = face[0][1] - round(face[0][2] / 6)
    x2 = face[0][0] + face[0][2] + round(face[0][2] / 48)
    y2 = face[0][1] + face[0][3] + round(face[0][2] / 6)

    cropped_image = resize_1[y1:y2, x1:x2]
    final = cv2.resize(cropped_image, (92, 112)) 
    
    return final








# -----------------------------------------------------------------------------------
#                                  Recognition 
# -----------------------------------------------------------------------------------

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


def asRowMatrix(X):
    if len(X) == 0:
        return np.array([])
    mat = np.empty((0, X[0].size), dtype=X[0].dtype)
    for row in X:
        mat = np.vstack((mat, np.asarray(row).reshape(1, -1)))
    return mat


def pca(X, y, num_components=0):
    n, d = X.shape
    if (num_components <= 0) or (num_components > n):
        num_components = n
    mu = X.mean(axis=0)
    X = X - mu
    if n > d:
        C = np.dot(X.T, X)
        eigenvalues, eigenvectors = np.linalg.eigh(C)
    else:
        C = np.dot(X, X.T)
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        eigenvectors = np.dot(X.T, eigenvectors)
        for i in range(n):
            eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[0:num_components].copy()
    eigenvectors = eigenvectors[:, 0:num_components].copy()
    return [eigenvalues, eigenvectors, mu]


def project(W, X, mu=None):
    if mu is None:
        return np.dot(X, W)
    return np.dot(X - mu, W)


class AbstractDistance(object):
    def __init__(self, name):
        self._name = name

    def __call__(self, p, q):
        raise NotImplementedError(" Every AbstractDistance must implement the __call__method.")

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return self._name


class EuclideanDistance(AbstractDistance):
    def __init__(self):
        AbstractDistance.__init__(self, " EuclideanDistance ")

    def __call__(self, p, q):
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        return np.sqrt(np.sum(np.power((p - q), 2)))


class CosineDistance(AbstractDistance):
    def __init__(self):
        AbstractDistance.__init__(self, " CosineDistance ")

    def __call__(self, p, q):
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        return -np.dot(p.T, q) / (np.sqrt(np.dot(p, p.T) * np.dot(q, q.T)))


class BaseModel(object):
    def __init__(self, X=None, y=None, dist_metric=EuclideanDistance(), num_components=0):
        self.dist_metric = dist_metric
        self.num_components = num_components
        self.projections = []
        self.class_centroids = {}
        self.avg_intra_dist = 1.0
        self.W = []
        self.mu = []
        if (X is not None) and (y is not None):
            self.compute(X, y)

    def compute(self, X, y):
        raise NotImplementedError(" Every BaseModel must implement the compute method.")

    def predict(self, X,return_confidence=False):
        """Retourne le label prédit et un score de ressemblance (0-1) si demandé.

        Le score compare la distance au centroïde de la classe prédite à la
        distance intra-classe moyenne observée à l'entraînement. Ce n'est pas
        une proba calibrée mais une mesure de ressemblance. Retourne aussi le
        ratio best/second-best pour juger la netteté de la décision.
        """
        minDist = np.finfo("float").max
        minClass = -1
        secondDist = np.finfo("float").max
        Q = project(self.W, X.reshape(1, -1), self.mu)
        for i, projection in enumerate(self.projections):
            dist = self.dist_metric(projection, Q)
            label = self.y[i]
            if dist < minDist:
                secondDist = minDist
                minDist = dist
                minClass = label
            elif dist < secondDist:
                secondDist = dist

        if not return_confidence:
            return minClass

        centroid = self.class_centroids.get(minClass)
        if centroid is None:
            return minClass, 0.0
        d_to_centroid = self.dist_metric(Q, centroid)
        scale = self.avg_intra_dist if self.avg_intra_dist > 1e-9 else 1.0
        similarity = 1.0 / (1.0 + (d_to_centroid / scale))
        similarity = float(np.clip(similarity, 0.0, 1.0))
        # Ratio best/second-best : plus il est grand, plus la décision est nette.
        ratio = secondDist / minDist if minDist > 0 else float("inf")
        return minClass, similarity, ratio


class EigenfacesModel(BaseModel):
    def __init__(self, X=None, y=None, dist_metric=EuclideanDistance(), num_components=0):
        super(EigenfacesModel, self).__init__(X=X, y=y, dist_metric=dist_metric, num_components=num_components)

    def compute(self, X, y):
        [D, self.W, self.mu] = pca(asRowMatrix(X), y, self.num_components)
        self.y = y
        for xi in X:
            self.projections.append(project(self.W, xi.reshape(1, -1), self.mu))
        # Calcule un centroïde par classe et la distance intra-classe moyenne
        label_to_projs = {}
        for proj, label in zip(self.projections, self.y):
            label_to_projs.setdefault(label, []).append(proj)
        for label, projs in label_to_projs.items():
            self.class_centroids[label] = np.mean(np.vstack(projs), axis=0)
        intra_sum = 0.0
        count = 0
        for label, projs in label_to_projs.items():
            centroid = self.class_centroids[label]
            for proj in projs:
                intra_sum += self.dist_metric(proj, centroid)
                count += 1
        self.avg_intra_dist = intra_sum / count if count else 1.0

def recongnition():
    # La constante data_root : “Chemin racine des données (à adapter si la base est déplacée).”

    print(f"Chargement de la base de visages depuis {dataset_path} ...")
    X, y, label_names = read_images(dataset_path)

    # Associe certains dossiers Sxx à des noms lisibles pour l'annonce.
    friendly_names = {
    }

    def describe_label(label_idx):
        if label_idx is None or label_idx < 0 or label_idx >= len(label_names):
            return "Inconnu", False
        raw = label_names[label_idx]
        alias = friendly_names.get(raw, raw)
        return alias, raw in friendly_names

    image_height, image_width = X[0].shape
    print(f"{len(X)} images chargées pour {len(label_names)} personnes.")
    print(f"Taille des images: {image_width}x{image_height} (LxH).")

    num_components = min(150, len(X)) #“Limite le nombre de composantes PCA pour éviter le surajustement et respecter le nombre d’images.”
    similarity_threshold = 0.30   # exige au moins 30% de ressemblance normalisée
    ratio_threshold = 1.03        # la meilleure distance doit être 3% plus faible que la 2e
    model = EigenfacesModel(X, y, num_components=num_components)

    target_size = (image_width, image_height)
    if not test_path.is_dir():
        raise ValueError(f"Dossier de test introuvable: {test_path}")
    # Charge chaque image test à la même taille que l’entraînement, projette et affiche la classe prédite.
    
    base = read_image(base_path)
    cropped = crop_image(base)
    
    cv2.imwrite(output_path, cropped)
    
    for img_path in sorted(p for p in test_path.iterdir() if p.is_file()):
        try:
            test_image = load_image_as_array(img_path, target_size=target_size)
        except Exception as exc:
            print(f"Lecture impossible pour {img_path}: {exc}")
            continue
        prediction, similarity, ratio = model.predict(test_image, return_confidence=True)
        is_strong_enough = (similarity >= similarity_threshold) and (ratio >= ratio_threshold)
        if not is_strong_enough:
            prediction = -1  # forcer "Inconnu"
        similarity_pct = round(similarity * 100, 1)
        predicted_name, is_friendly = describe_label(prediction)
        if is_friendly:
            print(f"{img_path.name} -> {predicted_name} détecté ! ({similarity_pct}% ressemblance)")
        else:
            print(f"{img_path.name} -> classe prédite {predicted_name} (id {prediction},{similarity_pct}% ressemblance)")







# -----------------------------------------------------------------------------------
#                                     Creation
# -----------------------------------------------------------------------------------

def creation_base(name):
    
    name = name.capitalize()
    # La constante data_root : “Chemin racine des données (à adapter si la base est déplacée).”

    base = read_image(base_path)
    cropped = crop_image_creation(base)
    
    person_dir = dataset_path / name
    
    if not person_dir.exists():
        print(f"Création d’un nouveau dossier : {person_dir}")
        person_dir.mkdir(parents=True, exist_ok=True)
        next_index = 1
    else:
        print(f"Dossier trouvé : {person_dir}")

        # Trouve les fichiers .pgm existants et calcule l’index suivant
        existing_files = [f for f in os.listdir(person_dir) if f.endswith(".pgm")]

        if existing_files:
            # Ex : "1.pgm" → 1
            indices = [int(f.split(".")[0]) for f in existing_files]
            next_index = max(indices) + 1
        else:
            next_index = 1
    
    output_path_creation = person_dir / f"{next_index}.pgm"
    
    cv2.imwrite(output_path_creation, cropped)
    print("Image sauvegardée dans :", output_path_creation)


if __name__ == "__main__":
    creation_base("tomy")
