from importlib.resources import path
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from arcface import ArcFace
import cv2
import sys 
import numpy as np 
import insightface 
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from PIL import Image
import warnings

app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))
face_rec = ArcFace.ArcFace()

class ArcfaceEmb():

    def __init__(self, path):
        self.path = path
        self.emb = self.get_embedding()
    
    def get_embedding(self):
        img = cv2.imread(self.path)
        if np.shape(img) == ():
            warnings.warn(f"Warning: image at{self.path} cannot be loaded. Skipping this image for now.")
        else:
            faces = app.get(img)
            face = faces[0].bbox.astype(np.int32)
            cropped = img[face[1]:face[3], face[0]:face[2]] 
            # cv2.imwrite("temp1.jpg", cropped1)
            # o1 = face_rec.calc_emb("temp1.jpg")
            emb = face_rec.calc_emb(cropped)
            return emb
    
    def get_embedding_dist(self, emb2):
        return face_rec.get_distance_embeddings(self.emb, emb2)
    



