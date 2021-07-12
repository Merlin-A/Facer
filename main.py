import face_recognition
import cv2
import numpy as np
import time


# Developed By Merlin (B-02)

print("\t \t ................Loading Please Wait..................")
time.sleep(1)
print("\t \t ................Loading Please Wait..................")
time.sleep(1)
print("\t \t ................Loading Please Wait..................")
time.sleep(1)



#This is to access the camera (0) is to specify the default webcam
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)


    









#---------------------------------------------------------------------------------------------------------------#
Sundar_Pichai_image = face_recognition.load_image_file('.\Images\Sundar_Pichai.jpg')
Sundar_Pichai_face_encoding = face_recognition.face_encodings(Sundar_Pichai_image)[0]

Jeff_Bezos_image = face_recognition.load_image_file('.\Images\Jeff_Bezos.jpg')
Jeff_Bezos_face_encoding = face_recognition.face_encodings(Jeff_Bezos_image)[0]

Elon_Musk_image = face_recognition.load_image_file('.\Images\Elon_Musk.jpg')
Elon_Musk_face_encoding = face_recognition.face_encodings(Elon_Musk_image)[0]


Modi_Beard_image = face_recognition.load_image_file('./Images/Modi1.jpg')
Modi_Beard_face_encoding = face_recognition.face_encodings(Modi_Beard_image)[0]

Modi_image = face_recognition.load_image_file('./Images/Modi.jpg')
Modi_face_encoding = face_recognition.face_encodings(Modi_image)[0]

Joe_Biden_image = face_recognition.load_image_file('./Images/Joe_Biden.jpg')
Joe_Biden_face_encoding = face_recognition.face_encodings(Joe_Biden_image)[0]

Satya_Nadella_image = face_recognition.load_image_file('./Images/Satya_Nadella.jpg')
Satya_Nadella_face_encoding = face_recognition.face_encodings(Satya_Nadella_image)[0]







#---------------------------------------------------------------------------------------------------------------------#



# Create arrays of known face encodings and their names
known_face_encodings = [
    Sundar_Pichai_face_encoding,
    Jeff_Bezos_face_encoding,
    Elon_Musk_face_encoding,
    Modi_face_encoding,
    Modi_Beard_face_encoding,
    Joe_Biden_face_encoding,
    Satya_Nadella_face_encoding
  
]
known_face_names = [
    "S. Pichai",
    "Jeff Bezos",
    "Elon Musk",
    "PM Modi",
    "PM Modi",
    "Joe Biden",
    "Satya Nadella"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


while True:
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1] #Indexing the format to change it's order

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

       
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 229, 0), 4)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (230, 222, 151), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255,255,255), 2)

    # Display the resulting image
    cv2.imshow('Facer', frame)

    

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# Release handle to the webcam
video_capture.release()

cv2.destroyAllWindows();
print("\n"*500);print("\n \t \t .................Closed Succesfully...................")