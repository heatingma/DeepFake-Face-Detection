import os
import pickle
import numpy as np
from tqdm import tqdm
from sklearn import svm
from skimage.io import imread
from skimage.feature import hog, local_binary_pattern, sift
from skimage.color import rgb2lab, rgb2gray
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class SVMClassifier:
    def __init__(
        self,
        kernel: str = "rbf",
        feature_type: str = "hog",
        data_dir: str = "homework_dataset/deep_face",
        pretrain: bool = False,
        pretrained_pickle_path: str = "checkpoint/svm_rbf.pkl",
        save_pickle_path: str = "checkpoint/svm.pkl"
    ):
        self.data_dir = data_dir
        self.train_data_dir = os.path.join(self.data_dir, "train")
        self.train_real_data_dir = os.path.join(self.train_data_dir, "real")
        self.train_fake_data_dir = os.path.join(self.train_data_dir, "fake")
        self.test_data_dir = os.path.join(self.data_dir, "test")
        self.test_real_data_dir = os.path.join(self.test_data_dir, "real")
        self.test_fake_data_dir = os.path.join(self.test_data_dir, "fake")
        
        # feature_type
        self.feature_type = feature_type
        
        # data
        self.images = None
        self.labels = None
        
        # model
        if pretrain:
            with open(pretrained_pickle_path, 'rb') as f:
                loaded_model = pickle.load(f)
            self.model = loaded_model
        else:
            self.model = svm.SVC(kernel=kernel)
        self.save_pickle_path = save_pickle_path
        
    def read_train_data(self):
        # fake
        for image_name in tqdm(os.listdir(self.train_fake_data_dir), desc="Reading Fake Images"):
            image_path = os.path.join(self.train_fake_data_dir, image_name)
            image = imread(image_path, as_gray=False)
            self.images.append(self.extract_feature(image))
            self.labels.append(0)
        
        # real
        for image_name in tqdm(os.listdir(self.train_real_data_dir), desc="Reading Real Images"):
            image_path = os.path.join(self.train_real_data_dir, image_name)
            image = imread(image_path, as_gray=False)
            self.images.append(self.extract_feature(image))
            self.labels.append(1)
    
    def read_test_data(self):
        # fake
        for image_name in tqdm(os.listdir(self.test_fake_data_dir), desc="Reading Fake Images"):
            image_path = os.path.join(self.test_fake_data_dir, image_name)
            image = imread(image_path, as_gray=False)
            self.images.append(self.extract_feature(image))
            self.labels.append(0)
        
        # real
        for image_name in tqdm(os.listdir(self.test_real_data_dir), desc="Reading Real Images"):
            image_path = os.path.join(self.test_real_data_dir, image_name)
            image = imread(image_path, as_gray=False)
            self.images.append(self.extract_feature(image))
            self.labels.append(1)
    
    def extract_feature(self, image: np.ndarray):
        image_gray = rgb2gray(image)

        if self.feature_type == "hog":
            # hog ()
            feature = hog(
                image_gray, pixels_per_cell=(32, 32), cells_per_block=(1, 1),
                orientations=9, block_norm='L2-Hys', visualize=False
            )
        elif self.feature_type == "color":
            # color_histogram ()
            lab_image = rgb2lab(image)
            color_histogram = []
            for channel in lab_image:
                hist, _ = np.histogram(channel, bins=3, range=(0, 100))
                color_histogram.append(hist)
            scaler = MinMaxScaler()
            feature = scaler.fit_transform(color_histogram).flatten()
        elif self.feature_type == "lbp":
            # lbp ()
            feature = np.mean(local_binary_pattern((
                image_gray*255).astype(np.int64), 8, 3, 'uniform'), axis=0
            ) / 255
        elif self.feature_type == "resize":
            # resize ()
            feature = resize(image, (3, 16, 16), anti_aliasing=True).flatten()
        
        return feature

    
    def train(self):
        # read data
        self.images = list()
        self.labels = list()
        self.read_train_data()
        self.images = np.array(self.images) # (B, 256, 256)
        self.labels = np.array(self.labels) # (B,)
        
        # normalize
        scaler = StandardScaler()
        self.images = scaler.fit_transform(self.images)
        
        # train
        self.model.fit(self.images, self.labels)
        
        # save_pickle_path
        with open(self.save_pickle_path, 'wb') as f:
            pickle.dump(self.model, f)
            
    def test(self):
        # read data
        self.images = list()
        self.labels = list()
        self.read_test_data()
        self.images = np.array(self.images) # (B, 1600)
        self.labels = np.array(self.labels) # (B,)
        
        # normalize
        scaler = StandardScaler()
        self.images = scaler.fit_transform(self.images)
        
        # test
        y_pred = self.model.predict(self.images)
        
        # ACC
        acc = accuracy_score(self.labels, y_pred) * 100
 
        # F1 score
        f1 = f1_score(self.labels, y_pred)* 100
        
        # AUC
        y_score = self.model.decision_function(self.images)
        auc = roc_auc_score(self.labels, y_score)* 100
        
        return acc, f1, auc