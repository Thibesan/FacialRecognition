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

                    self.faceLocations = face_recognition.face_locations(rgbSmallFrame)
                    self.faceEncodings = face_recognition.face_encodings(rgbSmallFrame, self.faceLocations)

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

                #Toggle Recognition Function, Preserve Resources
                self.processCurrentFrame = not self.processCurrentFrame

                #Display Function Info as Annotation
                for (top, right, bottom, left), name in zip(self.faceLocations, self.faceNames):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    #Rectangle Outline of Face Location
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    #Confidence Level Display in Outline
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                cv2.imshow('Face Recognition', frame)

                if cv2.waitKey(1) == ord('x'):
                    break

            videoCapture.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    fr = FaceRecognition()
    fr.runRecognition()


