from imutils.video import VideoStream
from imutils import face_utils
from gpiozero import Buzzer
from firebase import firebase
from datetime import datetime
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

DEVICE_ID = 'device0'
EMAIL_ADDR = 'tianyus2@illinois.edu'
firebase = firebase.FirebaseApplication('https://cs498-iot-project.firebaseio.com/', None)

def euclidean_dist(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def eye_aspect_ratio(eye):
    # Compute EAR Ratio. See the paper by Soukupova and Cech on
    # eye blinking detection algorithm
    
    # Compute distance between two sets of vertical points
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    
    # Compute the distance between a set pf horizontal points
    C = euclidean_dist(eye[0], eye[3])
    
    # Eye aspect ratio
    ear = (A + B) / (2.0 * C)
    
    return ear
    
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", default="./haarcascade_frontalface_default.xml",
    help = "path to where the face cascade resides")
ap.add_argument("-p", "--shape-predictor", default="./shape_predictor_68_face_landmarks.dat",
    help="path to facial landmark predictor")
args = vars(ap.parse_args())
    
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 16

COUNTER = 0
ALARM_ON = False

buzzer = Buzzer(17, active_high=False)

# Load OpenCV's Haar Cascade for face detector, then use dlib's
# shape predictor to compute facial landmarks
print("Loading detectors...")
detector = cv2.CascadeClassifier(args["cascade"])
predictor = dlib.shape_predictor(args["shape_predictor"])

lStart, lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
rStart, rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
# Start PiCamera video stream
print("Starting PiCamera...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(1.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect face
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
        minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    if len(rects) > 0:
        x, y, w, h = rects[0]
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        
        # Compute facial landmarks
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        ear = (leftEAR + rightEAR ) / 2
        
        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            # if the eyes were closed for a sufficient number of
            # frames, then sound the alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # if the alarm is not on, turn it on
                if not ALARM_ON:
                    ALARM_ON = True
                    buzzer.on()
                    post_data = {
                        'device': DEVICE_ID,
                        'time': datetime.now(),
                        'email': EMAIL_ADDR
                    }
                    firebase.post('/records', post_data)

                # draw an alarm on the frame
                cv2.putText(frame, "WARNING", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
              
            # draw text indicating eye status
            cv2.putText(frame, "Eye closed", (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the counter and alarm
        else:
            COUNTER = 0
            ALARM_ON = False
            buzzer.off()
            cv2.putText(frame, "Eye open", (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        print(ear)
        
    else:
        cv2.putText(frame, "No face detected", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
 
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop() 


    
