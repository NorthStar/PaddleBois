import numpy as np
import numpy
import cv2
import os.path
import urllib
import json
import requests
import scipy
from scipy import spatial
import tarfile,sys

#BACKEND_URL = "ip-172-31-42-171"
# 1
class word2vec:

    word_dict = dict()
    embedding_table = ""

    def __init__(self):
        self.load_files()
        self.word_dict = dict()
        with open("models/word2vec/word_dict", "r") as f:
            for line in f:
                key, value = line.strip().split(" ")
                self.word_dict[key] = value
        self.embedding_table = numpy.loadtxt("models/word2vec/embedding_table", delimiter=",")

    def run(self, s1, s2):
        if s1 in self.word_dict:
            print("contains s1")
        else:
            print(self.word_dict)
            print("doesn't contain")
        i1 = int(self.word_dict[s1])
        i2 = int(self.word_dict[s2])
        return spatial.distance.cosine(self.embedding_table[i1], self.embedding_table[i2])

    def load_files(self):
        if not os.path.isfile("models/word2vec/inference_topology.pkl"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/04.word2vec/inference_topology.pkl",
                               "models/word2vec/inference_topology.pkl")
            print("Loading models/word2vec/inference_topology.pkl . . .")

        if not os.path.isfile("models/word2vec/param.tar"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/04.word2vec/param.tar",
                               "models/word2vec/param.tar")
            extract_tar("models/word2vec/param.tar")
            print("Loading models/word2vec/param.tar . . .")

        if not os.path.isfile("models/word2vec/word_dict"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/04.word2vec/word_dict",
                               "models/word2vec/word_dict")
            print("Loading models/word2vec/word_dict . . .")

        if not os.path.isfile("models/word2vec/embedding_table"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/04.word2vec/embedding_table",
                               "models/word2vec/embedding_table")
            print("Loading models/word2vec/embedding_table . . .")

# 2
class image_classification:

    def __init__(self):
        self.load_files()


    def load_files(self):
        if not os.path.isfile("models/image_classification/inference_topology.pkl"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/03.image_classification/inference_topology.pkl",
                               "models/image_classification/inference_topology.pkl")
            print("Loading models/image_classification/inference_topology.pkl . . .")

        if not os.path.isfile("models/image_classification/param.tar"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/03.image_classification/param.tar",
                               "models/image_classification/param.tar")
            extract_tar("models/image_classification/param.tar")
            print("Loading models/image_classification/param.tar . . .")

    def run(self, img_file):
        BACKEND_URL = "http://35.167.14.53:8000"
        img = cv2.imread(img_file)
        print("read in image")
        img = np.swapaxes(img, 1, 2)
        img = np.swapaxes(img, 1, 0)
        arr = img.flatten()
        arr = arr / 255.0
        req = {"image": arr.tolist()}
        print(req)
        res = requests.request("POST", url=BACKEND_URL, json=req)
        print("printing json")
        print(json.dumps(res.json()))


# 3
class sentiment_classification:

    def __init__(self):
        self.load_files()

    def load_files(self):
        if not os.path.isfile("models/sentiment_classification/inference_topology.pkl"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/06.understand_sentiment/inference_topology.pkl",
                               "models/sentiment_classification/inference_topology.pkl")
            print("Loading models/sentiment_classification/inference_topology.pkl . . .")

        if not os.path.isfile("models/sentiment_classification/param.tar"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/06.understand_sentiment/param.tar",
                               "models/sentiment_classification/param.tar")
            extract_tar("models/sentiment_classification/param.tar")
            print("Loading models/sentiment_classification/param.tar . . .")

        if not os.path.isfile("models/sentiment_classification/word_dict.tar"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/06.understand_sentiment/word_dict.tar",
                               "models/sentiment_classification/word_dict.tar")
            extract_tar("models/sentiment_classification/word_dict.tar")
            print("Loading models/sentiment_classification/word_dict.tar . . .")
    
    def run(self, indices):
        BACKEND_URL = "http://35.167.14.53:3000"
        req = {"word":indices}
        print(req)
        res = requests.request("POST", url=BACKEND_URL, json=req)
        print(json.dumps(res.json()))

    

# 4
class machine_translation:

    def __init__(self):
        self.load_files()

    def load_files(self):
        if not os.path.isfile("models/machine_translation/inference_topology.pkl"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/08.machine_translation/inference_topology.pkl",
                               "models/machine_translation/inference_topology.pkl")
            print("Loading models/machine_translation/inference_topology.pkl . . .")

        if not os.path.isfile("models/machine_translation/param.tar"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/08.machine_translation/param.tar",
                               "models/machine_translation/param.tar")
            extract_tar("models/machine_translation/param.tar")
            print("Loading models/machine_translation/param.tar . . .")

        if not os.path.isfile("models/machine_translation/src_dict.txt"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/08.machine_translation/src_dict.txt",
                               "models/machine_translation/src_dict.txt")
            print("Loading models/machine_translation/src_dict.txt . . .")

        if not os.path.isfile("models/machine_translation/trg_dict.txt"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/08.machine_translation/trg_dict.txt",
                               "models/machine_translation/trg_dict.txt")
            print("Loading models/machine_translation/trg_dict.txt . . .")


# 5
class recognize_digits:

    def __init__(self):
        self.load_files()

    def load_files(self):
        if not os.path.isfile("models/recognize_digits/inference_topology.pkl"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/inference_topology.pkl",
                               "models/recognize_digits/inference_topology.pkl")
            print("Loading models/recognize_digits/inference_topology.pkl . . .")

        if not os.path.isfile("models/recognize_digits/param.tar"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/param.tar",
                              "models/recognize_digits/param.tar")
            extract_tar("models/recognize_digits/param.tar")
            print("Loading models/recognize_digits/param.tar . . .")


# 6
class object_detection:

    def __init__(self):
        self.load_files()

    def load_files(self):
        if not os.path.isfile("models/object_detection/inference_topology.pkl"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/SSD/inference_topology.pkl",
                               "models/object_detection/inference_topology.pkl")
            print("Loading models/object_detection/inference_topology.pkl . . .")

        if not os.path.isfile("models/object_detection/param.tar"):
            urllib.urlretrieve("https://s3.us-east-2.amazonaws.com/models.paddlepaddle/SSD/param.tar",
                              "models/object_detection/param.tar")
            extract_tar("models/object_detection/param.tar")
            print("Loading models/object_detection/param.tar . . .")


def extract_tar(file_path):
    tar = tarfile.open(file_path)
    tar.extractall()
    tar.close()
