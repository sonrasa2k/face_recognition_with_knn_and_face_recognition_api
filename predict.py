from numpy import ComplexWarning, byte
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import cv2
import time
from detect import return_box_faces
import math
def predict(X_img_path_or_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    if os.path.isfile(X_img_path_or_frame):
        X_img = face_recognition.load_image_file(X_img_path_or_frame)

    if not os.path.isfile(X_img_path_or_frame):
        X_img = X_img_path_or_frame

    # X_face_locations = face_recognition.face_locations(X_img)
    X_face_locations = return_box_faces(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
            zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
def show_prediction_labels_on_image(img_path, predictions):
    image = face_recognition.load_image_file(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    print(predictions)
    for name, (top, right, bottom, left) in predictions:
        top_right = (right, top)
        bottom_left = (left, bottom + 22)
        bottom_right = (right, bottom)
        a = left
        b = bottom - top
        top_left = (top, left)
        cv2.rectangle(image, top_right, bottom_left, (255, 0, 0), 3)
        cv2.putText(image, str(name), (left, bottom), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 0, 0), 1, cv2.FILLED)

    cv2.imshow(img_path, image)
    cv2.waitKey(0)
    cv2.destroyWindow(img_path)
if __name__ == '__main__':
    prediction = predict(X_img_path_or_frame="son_tung.jpg",model_path="model.clf")
    show_prediction_labels_on_image(img_path="son_tung.jpg",predictions=prediction)