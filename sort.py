
import cv2
import numpy as np
from mediapipe.tasks.python import vision
import mediapipe as mp
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import os

from filterpy.kalman import KalmanFilter
from tqdm import tqdm



# Naming a window
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

# Using resizeWindow()
cv2.resizeWindow("Frame", 700, 400)


class Model():
    def __init__(self, detection_model_filepath = "model/object_detector/efficientdet_lite0_uint8.tflite", embedding_model_filepath="model/embedder/mobilenet_v3_small_075_224_embedder.tflite"):
        self.objectDectector_model_filepath = detection_model_filepath
        # video_filepath = "dataset/video_cars.mp4"
        self.embedder_model_filepath = embedding_model_filepath 

        self.initializeObjDetModel()

        self.initializeEmbeddingModel()
    
    def initializeObjDetModel(self):
        self.ObjecttDetectorOptions = mp.tasks.vision.ObjectDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.objectDectector_model_filepath),
            max_results=50,
            score_threshold=0.4,
            running_mode=mp.tasks.vision.RunningMode.VIDEO
            # running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM
        )

    def initializeEmbeddingModel(self):
        # embeddings:
        # Create options for Image Embedder
        embedder_base_options = mp.tasks.BaseOptions(model_asset_path=self.embedder_model_filepath)
        l2_normalize = True #@param {type:"boolean"}
        quantize = True #@param {type:"boolean"}
        self.emb_options = vision.ImageEmbedderOptions(
            base_options=embedder_base_options,
            l2_normalize=l2_normalize, 
            quantize=quantize,
            running_mode= mp.tasks.vision.RunningMode.VIDEO
            # running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM
            )
        
class Track():
    def __init__(self, track_idx, bbox, features, frame_limit =10, input_st=7, output_st=4):
        # input_st = 7
        # output_st = 4
        
        self.warming = True
        self.warming_count = 5 # that many frames needed to predict accuratly for the pose
        self.set_kf(input_st, output_st)
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)

        self.track_idx = track_idx
        self.obj_not_seen = 0 # object not seen for certain frames count
        self.frame_limit = frame_limit
        self.features = features

        self.time_since_update = 0

    def notFound(self):
        _ = self.predict()
        
        if self.time_since_update>self.frame_limit:
            return True
        return False

    def set_kf(self, input_st, output_st):
        self.kf = KalmanFilter(dim_x=input_st, dim_z=output_st)

        # A
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])

        # H
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

    def convert_bbox_to_z(self,bbox):
        """
        Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h    #scale is just area
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    def convert_x_to_bbox(self,x):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    
    def update(self,bbox, features):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        history = []

        # update features
        prev_feature = self.features.embedding
        new_feature = features.embedding
        self.features.embedding = (prev_feature+new_feature)/2

        self.kf.update(self.convert_bbox_to_z(bbox))
        if self.warming:
            self.warming_count-=1
            if self.warming_count<=0:
                self.warming=False

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.time_since_update += 1
        # history.append()
        return self.kf.x #self.convert_x_to_bbox(self.kf.x)
        # return history[-1]

class Sort:
    def __init__(self, video_filepath = "dataset/video_cars.mp4",
                 detection_model_filepath = "model/object_detector/efficientdet_lite0_uint8.tflite",
                 embedding_model_filepath="model/embedder/mobilenet_v3_small_075_224_embedder.tflite",
                 visualize=True,
                 save_output=True):
        self.cap = cv2.VideoCapture(video_filepath)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print("FPS: ",self.fps)

        # setting up video writer for testing the object detector
        # We need to set resolutions.
        # so, convert them from float to integer.
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        
        size = (frame_width, frame_height)
        
        ## initialize the models
        self.models = Model(detection_model_filepath=detection_model_filepath,embedding_model_filepath=embedding_model_filepath)

        #setting parameters
        self.iou_thresh = 0.7
        self.feature_matching_thresh = 0.65
        self.car_idx = 0 #car counts detected cars in the video, same car will be not counted more than 1
        self.obj_track = {} # keep the object tracking data 

        self.visualize = visualize
        self.save_output = save_output

        if save_output:
            # Below VideoWriter object will create
            # a frame of above defined The output 
            # # is stored in 'filename.avi' file.
            self.videoWriter = cv2.VideoWriter('objEmbed_video_cars_track.avi', 
                                    cv2.VideoWriter_fourcc(*'MJPG'),
                                    10, size)
            self.output_image_dir = "images"
            os.makedirs(self.output_image_dir,exist_ok=True)
        
        

    def IOU(self,bbox1,bbox2):
        
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(bbox1[0], bbox2[0])
        yA = max(bbox1[1], bbox2[1])
        xB = min(bbox1[2], bbox2[2])
        yB = min(bbox1[3], bbox2[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        bbox1Area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
        bbox2Area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(bbox1Area + bbox2Area - interArea)
        # return the intersection over union value
        return iou
    
    def run(self):
        frame_idx = 0
        # Embedding with tracking
        # Create Image Embedder
        crop_idx = 0

        all_detections=defaultdict(list)
        avg_processing_time = 0
        prev_embeddings = [] # prev frame as 1, next frame as 2
        next_embeddings = []

        frame_interval = 1

        with vision.ImageEmbedder.create_from_options(self.models.emb_options) as embedder:
            with mp.tasks.vision.ObjectDetector.create_from_options(self.models.ObjecttDetectorOptions) as detector:
                while self.cap.isOpened():
                    ret, img = self.cap.read()
                    if ret == True:

                        frame_idx+=1
                        print("Processing frame : ",frame_idx)
                        # Calculate the timestamp of the current frame
                        frame_timestamp_ms = int(1000 * frame_idx / self.fps)
                        # print(frame_timestamp_ms)
                        time_start = time.perf_counter()
                        # if frame_idx%frame_interval==0:
                        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

                        # Perform object detection on the video frame.
                        detection_results = detector.detect_for_video(mp_image, frame_timestamp_ms)

                        dets = []
                        for det_res in detection_results.detections:
                            x = det_res.bounding_box.origin_x
                            y = det_res.bounding_box.origin_y
                            width = det_res.bounding_box.width
                            height = det_res.bounding_box.height
                            
                            x1,y1,x2,y2 = (x,y,x+width,y+height)
                            dets.append((x1,y1,x2,y2))

                            img_crop = img[y1:y2,x1:x2,:].astype(np.uint8)
                            crop_idx+=1
                            # Calculate the timestamp of the current frame
                            frame_timestamp_ms = int(1000 * crop_idx / self.fps)
                            # Convert the frame received from OpenCV to a MediaPipe’s Image object.
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_crop)

                            # Perform object detection on the video frame.
                            embedding_result = embedder.embed_for_video(mp_image, frame_timestamp_ms)

                            # embed_arr.append(embedding_result.embeddings[0])
                            features = embedding_result.embeddings[0]
                            bbox = [x1,y1,x2,y2]
                            all_detections[frame_idx].append((bbox, features))
                            # img = cv2.rectangle(img, (x,y),(x+width,y+height),color=(0,255,0),thickness=2)
                        
                            if len(prev_embeddings)==0:

                                self.car_idx+=1

                                # embed_arr.append(embedding_result.embeddings[0])
                                # car_idx_arr.append(car_idx)
                                self.obj_track[self.car_idx] = self.obj_track.get(self.car_idx, Track(self.car_idx,bbox=bbox, features=features))

                                # print(int(frame_idx//frame_interval),embedding_result)
                                prev_embeddings.append((features,self.car_idx,bbox))
                            # print(obj_track)
                                # break
                            else:

                                em1 = features
                                best_match = 0
                                new_car_idx = None
                                idx_remove = None
                                best_iou = 0
                                best_iou_idx = 0
                                for idx, (embed_prev, c_idx, coords) in enumerate(prev_embeddings):
                                    em2 = embed_prev
                                    similarity = vision.ImageEmbedder.cosine_similarity(
                                        em1,
                                        em2)
                                    
                                    iou = self.IOU(bbox, coords)
                                    if iou>self.iou_thresh and iou>best_iou:
                                        best_iou = iou
                                        best_iou_idx = idx

                                    if similarity>best_match:
                                        # print(similarity)
                                        best_match = similarity
                                        if similarity>self.feature_matching_thresh:
                                            
                                            new_car_idx = c_idx
                                            idx_remove = idx
                                if new_car_idx is None:
                                    flag= True
                                    print("Similarty : ",best_match)
                                    if best_iou_idx!=0:
                                        bbox_p = self.obj_track[best_iou_idx].predict().squeeze().tolist()
                                        iou = self.IOU(bbox_p,bbox)
                                        # print(iou)
                                        if iou>0.95:
                                            # print("Got data but embeddings are not matching")
                                            # obj_track[best_iou_idx].predict()
                                            self.obj_track[best_iou_idx].update(bbox=bbox,features=features)
                                            flag= False
                                            new_car_idx = best_iou_idx
                                            
                                    if flag:
                                        self.car_idx+=1
                                        new_car_idx = self.car_idx
                                        self.obj_track[self.car_idx] = self.obj_track.get(self.car_idx, Track(self.car_idx, bbox=bbox, features=features))
                                else:
                                    prev_embeddings.pop(idx_remove)
                                    x1,y1,x2,y2 = bbox
                                    pred_box = self.obj_track[new_car_idx].predict()
                                    yield pred_box
                                    # print("actual bbox : ",bbox," | pred bbox : ",pred_box)

                                    self.obj_track[new_car_idx].update(bbox=bbox,features=features)
                                # print(int(frame_idx//frame_interval),embedding_result)
                                next_embeddings.append((features,new_car_idx,bbox))
                        # reduce the count of predict for the cars not found
                        for _,c_id,_ in prev_embeddings:
                            ret = self.obj_track[c_id].notFound()
                            if ret:
                                # object not found for certain frames deleting the object
                                del self.obj_track[c_id]
                        prev_embeddings = next_embeddings.copy()
                        next_embeddings = []
                        # print(prev_embeddings)
                        if self.visualize or self.save_output:
                            for embed_prev, c_idx, coords in prev_embeddings:
                                x1 = coords[0]
                                y1 = coords[1]
                                x2 = coords[2]
                                y2 = coords[3]
                                img = cv2.rectangle(img, (x1,y1-100),(x1+100,y1),color=(255,0,0),thickness=-1)
                                img = cv2.putText(img, f'{c_idx}', org=(x1,y1),
                                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                                    fontScale=3,
                                                    color=(255,255,255),
                                                    thickness=4,
                                                    lineType=cv2.LINE_AA)
                                img = cv2.rectangle(img, (x1,y1),(x2,y2),color=(0,255,0),thickness=5)
                        if self.save_output:
                            cv2.imwrite(os.path.join(self.output_image_dir,f"image_{frame_idx}.png"),img)
                            self.videoWriter.write(img)
                        if self.visualize:
                            cv2.imshow("Frame", img)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break

                        try:    
                            avg_processing_time+=(time.perf_counter()-time_start)/((frame_idx-1)//frame_interval)
                        except ZeroDivisionError as ZE:
                            avg_processing_time+=(time.perf_counter()-time_start)

                        time_processing = time.perf_counter()-time_start
                        print("time take to process frame : ", time_processing)
                        avg_processing_time += time_processing
                    else:
                        print("Not able to read the frame")
                        break

        self.cap.release()
        if self.save_output:
            self.videoWriter.release()
        print("time taken : ",avg_processing_time/frame_idx)


if __name__ == "__main__":
    deep_sort = Sort(video_filepath=0)
    deep_sort.run()