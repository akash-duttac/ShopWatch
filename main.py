import csv
import cv2
import argparse
import numpy as np
import cvlib as cv
import uuid
import os
import datetime
import tensorflow
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
from keras.models import load_model
from transformers import ViTFeatureExtractor, ViTForImageClassification
from imgbeddings import imgbeddings
from sklearn.metrics.pairwise import cosine_similarity


csv_file = 'data/data.csv'
time_file = 'data/time.csv'

# Define a similarity threshold
SIMILARITY_THRESHOLD = 0.9

# Assuming 5 seconds for end time
END_TIME_THRESHOLD_SECONDS = 5

# Global variables
embedding_id_map = {}
embeddings_list = []  # List to store all embeddings for similarity comparison
uid_start_time_map = {}  # Map to store start time for each uid


# Load the TensorFlow SavedModel as an inference-only layer
model = load_model('gender_detection.model')
ibed = imgbeddings()
classes = ['male', 'female']

# Load the ViT Age Detection Model
age_model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
age_transforms = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')
age_classes = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]


def gender_detection(f, frame) :        
    #get corner points of face rectangle
    (startX, startY) = f[0], f[1]
    (endX, endY) = f[2], f[3]
    #crop the detected face region
    face_crop = np.copy(frame[startY:endY, startX:endX])

    if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
        return "NA"

    #preprocessing for gender detection model
    face_crop = cv2.resize(face_crop, (96, 96))
    face_crop = face_crop.astype("float") / 255.0
    face_crop = np.expand_dims(face_crop, axis=0)

    #apply gender detection on face
    conf = model.predict(face_crop)[0]
    
    #get label with maximum accuracy
    idx = np.argmax(conf)
    return classes[idx]

def age_detection(face, frame):
    startX, startY, endX, endY = face
    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    face_crop = np.copy(frame[startY:endY, startX:endX])

    # Using https://huggingface.co/nateraw/vit-age-classifier
    inputs = age_transforms(face_crop, return_tensors='pt')
    output = age_model(**inputs)
    proba = output.logits.softmax(1)
    preds = proba.argmax(1)
    age_group = age_classes[preds[0]]
    return age_group

def load_existing_embeddings(csv_file, time_file, section_name):
    embedding_id_map = {}
    embeddings_list = []
    uid_start_end_time_map = {}  # Map to store start and end time for each UID

    # Load embeddings from data.csv
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                uid, embedding_str, gender, age = row[:4]  # Extract the first 5 columns
                embedding = np.array(eval(embedding_str))  # Convert string representation back to numpy array
                embedding_id_map[len(embeddings_list)] = uid
                embeddings_list.append(embedding)

    # Load time information from time.csv
    if os.path.exists(time_file):
        with open(time_file, 'r') as timefile:
            time_reader = csv.reader(timefile)
            for row in time_reader:
                # Ensure row has enough values before unpacking
                if len(row) >= 4:
                    uid, section, start_time_str, end_time_str = row[:4]
                    start_time = datetime.datetime.strptime(start_time_str.strip(), "%Y-%m-%d %H:%M:%S.%f")
                    end_time = datetime.datetime.strptime(end_time_str.strip(), "%Y-%m-%d %H:%M:%S.%f")
                    uid_start_end_time_map[uid] = {'section': section, 'start_time': start_time, 'end_time': end_time}

    return embedding_id_map, embeddings_list, uid_start_end_time_map


def calculate_cosine_similarity(new_embedding, current_time, section):
    if len(embeddings_list) == 0:
        return 0.0  # Return 0 if no embeddings are available
    similarities = cosine_similarity([new_embedding], embeddings_list)
    max_similarity = np.max(similarities)
    
    # Check for existing embeddings and update end time if needed
    for idx, similarity in enumerate(similarities[0]):
        if idx not in embedding_id_map:
            continue  # Skip if the index is not in the map
        existing_uid = embedding_id_map[idx]
        if existing_uid not in uid_start_time_map:
            continue  # Skip if start and end time are not available for the UID
        elapsed_time_seconds = (current_time - uid_start_time_map[existing_uid]['start_time']).total_seconds()
        if similarity >= SIMILARITY_THRESHOLD and elapsed_time_seconds > END_TIME_THRESHOLD_SECONDS:
            if section == uid_start_time_map[existing_uid]['section']:
                # Update end time if conditions are met
                uid_start_time_map[existing_uid]['end_time'] = current_time
                with open(time_file, 'a', newline='') as timefile:
                    time_writer = csv.writer(timefile)
                    time_writer.writerow([existing_uid, section, uid_start_time_map[existing_uid]['start_time'], current_time])
    return max_similarity

def createImgbeddings(face, frame, section_name, current_time):
    global csv_file, time_file, embedding_id_map, embeddings_list, uid_start_time_map
    embedding_id_map, embeddings_list, uid_start_time_map = load_existing_embeddings(csv_file, time_file, section_name)

    with open(csv_file, 'a', newline='') as csvfile, open(time_file, 'a', newline='') as timefile:
        csv_writer = csv.writer(csvfile)
        time_writer = csv.writer(timefile)

        for start_x, start_y, end_x, end_y in face:
            if end_y < start_y + 10 or end_x < start_x + 10:
                continue
            embedding = ibed.to_embeddings(
                Image.fromarray(frame[start_y:end_y, start_x:end_x])
            )[0]
            string_representation = "[" + ",".join(str(x) for x in embedding.tolist()) + "]"

            uid = "unknown"
            age = "unknown"
            gender = "unknown"

            # Check if embedding already exists
            similarity = calculate_cosine_similarity(embedding, current_time, section_name)
            if similarity >= SIMILARITY_THRESHOLD:
                # Find the corresponding ID based on highest similarity
                max_similarity_index = np.argmax(similarity)
                uid = embedding_id_map[max_similarity_index]
                if uid in uid_start_time_map and section_name != uid_start_time_map[uid]['section']:
                    # Update end time and section in CSV
                    end_time = current_time
                    time_writer.writerow([uid, section_name, uid_start_time_map[uid]['start_time'], end_time])
                    uid_start_time_map[uid] = {'start_time': current_time, 'section': section_name}
            else:
                uid = str(uuid.uuid4())
                embedding_id_map[len(embeddings_list)] = uid
                embeddings_list.append(embedding)
                uid_start_time_map[uid] = {'start_time': current_time, 'section': section_name}
                gender = gender_detection((start_x, start_y, end_x, end_y), frame)
                age = age_detection((start_x, start_y, end_x, end_y), frame)
                csv_writer.writerow([uid, string_representation, gender, age])  # Write to data.csv
                time_writer.writerow([uid, section_name, current_time, current_time])  # Start and end time are the same for new entries
                print("New Person Inserted in data.csv and time.csv")

            cv2.putText(frame, f"ID: {uid[:6]}", (start_x, start_y - 20),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), thickness=2,)
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 51, 28), thickness=1,)


def process_frame(frame, section_name):
    face_detection_result = cv.detect_face(frame)

    # Check if faces were detected
    if face_detection_result is not None:
        faces, _ = face_detection_result 
        current_time = datetime.datetime.now()
        createImgbeddings(faces , frame, section_name, current_time)

def main():
    parser = argparse.ArgumentParser(description="Minor Project")     # Setting up Web-Cam Resolution
    parser.add_argument("section", type=str, help="Section where the camera is located")
    parser.add_argument("--webcam-resolution", default=[640, 480], nargs=2, type=int)
    args = parser.parse_args()
    section_name = args.section

    frame_width, frame_height = args.webcam_resolution


    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT , frame_height)

    # Initializing object tracker
    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            continue
        process_frame(frame, section_name)
        cv2.imshow(f"CCTV - {section_name}", frame)
        # Detect faces
        if cv2.waitKey(30) == 27:  
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()