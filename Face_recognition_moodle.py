# Facial recognition with Eigenfaces
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy as np
from PIL import Image


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
        self.W = []
        self.mu = []
        if (X is not None) and (y is not None):
            self.compute(X, y)

    def compute(self, X, y):
        raise NotImplementedError(" Every BaseModel must implement the compute method.")

    def predict(self, X):
        minDist = np.finfo("float").max
        minClass = -1
        Q = project(self.W, X.reshape(1, -1), self.mu)
        for i in range(len(self.projections)):
            dist = self.dist_metric(self.projections[i], Q)
            if dist < minDist:
                minDist = dist
                minClass = self.y[i]
        return minClass


class EigenfacesModel(BaseModel):
    def __init__(self, X=None, y=None, dist_metric=EuclideanDistance(), num_components=0):
        super(EigenfacesModel, self).__init__(X=X, y=y, dist_metric=dist_metric, num_components=num_components)

    def compute(self, X, y):
        [D, self.W, self.mu] = pca(asRowMatrix(X), y, self.num_components)
        self.y = y
        for xi in X:
            self.projections.append(project(self.W, xi.reshape(1, -1), self.mu))


# “Charge la base Base_Visages, entraîne Eigenfaces,
#  puis parcourt Images_de_test pour prédire et afficher les labels.”


def main():
    # La constante data_root : “Chemin racine des données (à adapter si la base est déplacée).”
    data_root = Path("/Users/chaterbach-char/Desktop/sni-5/Données_PROJ942")
    dataset_path = data_root / "Base_Visages"
    test_path = data_root / "Images_de_test"

    print(f"Chargement de la base de visages depuis {dataset_path} ...")
    X, y, label_names = read_images(dataset_path)
    image_height, image_width = X[0].shape
    print(f"{len(X)} images chargées pour {len(label_names)} personnes.")
    print(f"Taille des images: {image_width}x{image_height} (LxH).")

    num_components = min(150, len(X)) #“Limite le nombre de composantes PCA pour éviter le surajustement et respecter le nombre d’images.”
    model = EigenfacesModel(X, y, num_components=num_components)

    def describe_label(label_idx):
        return label_names[label_idx] if 0 <= label_idx < len(label_names) else str(label_idx)

    target_size = (image_width, image_height)
    if not test_path.is_dir():
        raise ValueError(f"Dossier de test introuvable: {test_path}")
    
#Charge chaque image test à la même taille que l’entraînement, projette et affiche la classe prédite.
    for img_path in sorted(p for p in test_path.iterdir() if p.is_file()):
        try:
            test_image = load_image_as_array(img_path, target_size=target_size)
        except Exception as exc:
            print(f"Lecture impossible pour {img_path}: {exc}")
            continue
        prediction = model.predict(test_image)
        predicted_name = describe_label(prediction)
        print(f"{img_path.name} -> classe prédite {predicted_name} (id {prediction})")


if __name__ == "__main__":
    main()
