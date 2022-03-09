# Count how many predictions we have done
from tensorflow.python.keras.models import load_model
from data.keypoints import mp_holistic,mediapipe_detection, draw_styled_landmarks
from data.datarecord import actions
from data.values import extract_keypoints
from predict.pronostico import  predice, extraerkeypoints
from predict.datapoint import get_datapoint
import numpy as np
import cv2
global counter
counter = 0
global sequence2
sequence2 = []

def inference(frame):
   
    sentence  = get_datapoint(frame)    
    global counter
    counter = counter + 1
        
    return f"player did  {sentence}{counter}"

