import face_recognition
import os, sys
import cv2
import numpy as np
import math

#Accuracy of face recognition
def faceConfidence(faceDistance, faceMatchThreshold = 0.6):
    range = (1.0 - faceMatchThreshold)
    linearVal = (1.0 - faceDistance) / (range * 2.0)

    #Linear Value to Percentage Conversion for Accuracy Representation
    if faceDistance < faceMatchThreshold:
        return str(round(linearVal * 100, 2)) + "%"
    else:
        value = (linearVal + ((1.0 - linearVal) *  math.pow((linearVal - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + "%"
    
class FaceRecognition:
    faceLocations = []
    faceEncodings = []
    faceNames = []
    knownFaceEncodings = []
    knownFaceNames = []
    processCurrentFrame = True #Limit processing

    def __init__(self):
        self.encodeFaces()
    #Encode Faces
    def encodeFaces(self):
        for image in os.listdir('faces'):
            #Load Image, encode it, and add to list of known faces w/ title
            faceImage = face_recognition.load_image_file(f"faces/{image}")
            #Will throw error if image encoding fails to recognize face
            faceEncoding = face_recognition.face_encodings(faceImage)[0]
            
            self.knownFaceEncodings.append(faceEncoding)
            self.knownFaceNames.append(image)
        
        print(self.knownFaceNames)

        def runRecognition(self):
            videoCapture = cv2.VideoCapture(0)

            if not videoCapture.isOpened():
                sys.exit("Video source not found")

            while True:
                ret, frame = videoCapture.read()

                if self.processCurrentFrame:
                    smallFrame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgbSmallFrame = cv2.cvtColor(smallFrame, cv2.COLOR_BGR2RGB) #Change Color params for 1:4 scale resized frame

                    self.faceLocations = face_recognition.faceLocations(rgbSmallFrame)
                    self.faceEncodings = face_recognition.faceEncodings(rgbSmallFrame, self.faceLocations)

                    self.faceNames = []
                    for faceEncoding in self.faceEncodings:
                        matches = face_recognition.compare_faces(self.knownFaceEncodings, faceEncoding)
                        name = 'Unknown'
                        confidence = 'Unknown'

                        faceDistances = face_recognition.face_distance(self.knownFaceEncodings, faceEncoding)
                        bestMatchIndex = np.argmin(faceDistances)

                        if matches[bestMatchIndex]:
                            name = self.knownFaceNames[bestMatchIndex]
                            confidence = faceConfidence(faceDistances[bestMatchIndex])

                        self.faceNames.append(f'{name} ({confidence})')

                self.processCurrentFrame = not self.processCurrentFrame

if __name__ == '__main__':
    fr = FaceRecognition()


