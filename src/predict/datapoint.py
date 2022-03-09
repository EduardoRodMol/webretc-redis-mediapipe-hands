from data.keypoints import mp_holistic,mediapipe_detection, draw_styled_landmarks
from data.datarecord import actions
from data.values import extract_keypoints
from predict.pronostico import  predice, extraerkeypoints

global sequence2 
sequence2 = []

def get_datapoint(frame):

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        global sequence2         
        sequence2.append(extraerkeypoints(frame,holistic))
        sequence2 = sequence2[-30:]
        sentence= "getting info"
        if len(sequence2) == 30:                    
            sequence2 = []
            sentence = predice(sequence2)
            print("Running inference...")

    return sentence