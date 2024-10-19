"""
This Python script defines a class `Predictor` for image classification using a pre-trained EfficientNet model. It also includes functions for detecting facial bounding boxes, reading and deleting image vectors from a database, and a main block for real-time face detection and recognition from a video source. 

The `Predictor` class is initialized with an EfficientNet model fine-tuned for a specific number of output features. It provides a method `predict_cls` to classify an input image.

The `detect_bounding_box` function detects faces in a video frame using a Haar cascade classifier. The `read_database` function reads image vectors from a specified directory, and `delete_database` removes all files from the same directory.

The script uses OpenCV for image processing and real-time video capture. Detected faces are cropped, converted to grayscale, classified, and compared against saved vectors in the database using cosine similarity. If a face doesn't match any existing vectors above a certain threshold, it is saved to the database.

The main block continuously processes video frames, detects faces, performs recognition, and updates the database accordingly. It also handles periodic database resets based on time conditions.

Note: The code snippet provided is missing some necessary imports and definitions such as `face_classifier`, functions like `effecientnet_model`, and `EfficientNet_B0_Weights`. Ensure these are defined elsewhere for the script to run correctly.

"""


import torch 
from torchvision import transforms as T 
import cv2 
import os 
from torchvision.models import efficientnet_b0 as effecientnet_model
from  torchvision.models import EfficientNet_B0_Weights
from torch import nn 
from datetime import datetime 
import numpy as np



class Predictor():

    def __init__(self, n_features=1000):
        super().__init__()


        model = effecientnet_model(EfficientNet_B0_Weights)    
        n_inputs = model.classifier[1].in_features

        model.classifier[1] = nn.Linear(n_inputs,n_features)
        
        model.cuda()

        self.model = model.eval()


        self.transforms = T.Compose([T.ToTensor(),
                       T.Resize((240,240)),
                       T.Normalize(( 0.5), ( 0.5))
                       
                       ])
        


    def predict_cls(self,image): 
        image = np.repeat(image[..., np.newaxis], 3, -1)
        image = self.transforms(image.copy())
        image = torch.unsqueeze(image, 0)

        # set the module to evaluation mode
        with torch.no_grad():


            # move data to GPU
            if torch.cuda.is_available():
                image = image.cuda()

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            logits  = self.model(image) # YOUR CODE HERE


        return logits.squeeze()
    

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    if len(faces): 
        for (x, y, w, h) in faces:
            pass 
            # cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        return faces, (x,y,h,w)
    else: 
        return False , None  



def read_database(database_path): 

    file_names = os.listdir(database_path)

    vectors = [torch.load(os.path.join(database_path, file_name)) for file_name in file_names]

    return vectors



def delete_database(database_path):
    
    file_names = os.listdir(database_path)

    [os.remove(os.path.join(database_path, file_name)) for file_name in file_names]

    return  





if __name__ == "__main__": 


    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    predictor = Predictor(n_features= 128)
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    video_capture = cv2.VideoCapture("faces_multi.mp4")

    database_path = "database/images"

    vecotors_path = "database/vectors"
    delete_database(vecotors_path)


    matching_threshold = 0.5
    counter = 0 
    


    while True:
        current_hour = datetime.now().hour
        current_minute = datetime.now().minute
        if current_hour in [12, 24] and current_hour> last_hour and  current_minute==0: 
            counter = 0 
            delete_database(vecotors_path)

        print(counter)
        if counter < 10: 
            ret, bgr_frame = video_capture.read()  # read frames from the video

            
            if ret is False:
                break  # terminate the loop if the frame is not read successfully

            faces, cords = detect_bounding_box(
                bgr_frame
            )  # apply the function we created to the video frame
            if cords is None: 
                continue
            
            x, y, w, h = cords

            crop = bgr_frame[y:y+h, x:x+w, :] 
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            vector  = predictor.predict_cls(crop)



        
            current_vectors = read_database(vecotors_path)

            scores = [cos(vector, saved_vector) for saved_vector in current_vectors]
            
            if len(scores) ==  0: 
                torch.save(vector, os.path.join(vecotors_path, f"{counter}.pt"))
                cv2.imwrite(os.path.join(database_path, f"{counter}.png"), crop)
                counter += 1 
                print("continuibg")
                continue 
                
            max_score = max(scores)
            print(max_score)
            
            if max_score < matching_threshold: 
                torch.save(vector, os.path.join(vecotors_path, f"{counter}.pt"))
                cv2.imwrite(os.path.join(database_path, f"{counter}.png"), crop)
                counter += 1 

            


            

        
                

                
        last_hour = current_hour

        cv2.imshow(
            "My Face Detection Project", crop
        )  # display the processed frame in a window named "My Face Detection Project"

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    video_capture.release()
    cv2.destroyAllWindows()

