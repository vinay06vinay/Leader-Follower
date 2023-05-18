#!/usr/bin/env python


import cv2
import numpy as np
from mediapipe.tasks.python import vision
import mediapipe as mp
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import os

from copy import copy
from filterpy.kalman import KalmanFilter
from tqdm import tqdm
import argparse


# Naming a window
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

# Using resizeWindow()
cv2.resizeWindow("Frame", 700, 400)


class Model():
    def __init__(self, 
                 detection_model_filepath = "model/object_detector/efficientdet_lite0_uint8.tflite", 
                 embedding_model_filepath="model/embedder/mobilenet_v3_large.tflite"):
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
        quantize = False #@param {type:"boolean"}
        self.emb_options = vision.ImageEmbedderOptions(
            base_options=embedder_base_options,
            l2_normalize=l2_normalize, 
            quantize=quantize,
            running_mode= mp.tasks.vision.RunningMode.VIDEO
            # running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM
            )
        
class Track():
    def __init__(self, track_idx, bbox, features, frame_limit =30, input_st=7, output_st=4):
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
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.],dtype=np.uint8).reshape((4,)).tolist()
    
    def update(self,bbox, features):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        history = []

        # update features
        prev_feature = self.features.embedding
        new_feature = features.embedding
        # print(prev_feature.dtype)
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

    def get_bbox(self):
        return self.convert_x_to_bbox(self.kf.x)

class Sort:
    def __init__(self, video_filepath = "dataset/video_cars.mp4",
                 detection_model_filepath = "model/object_detector/efficientdet_lite0_uint8.tflite",
                 embedding_model_filepath="model/embedder/mobilenet_v3_large.tflite",
                 frame_limit = 60,
                 size=(640,480),
                 output_video='objEmbed_video_cars_track.avi',
                 visualize=True,
                 save_output=True):
       
        ## initialize the models
        self.models = Model(detection_model_filepath=detection_model_filepath,embedding_model_filepath=embedding_model_filepath)
        self.embedder = vision.ImageEmbedder.create_from_options(self.models.emb_options)
        
        #setting parameters
        self.frame_limit = frame_limit # after this frames the track id will be remove from the memory
        self.iou_thresh = 0.7
        self.feature_matching_thresh = 0.65
        self.track_idx = 0 #car counts detected cars in the video, same car will be not counted more than 1
        self.obj_track = {} # keep the object tracking data 

        ## for track function 
        self.frame_idx = 0

        self.visualize = visualize
        self.save_output = save_output
        self.output_video = output_video

        self.video_filepath = video_filepath

        if self.save_output:
            # Below VideoWriter object will create
            # a frame of above defined The output 
            # # is stored in 'filename.avi' file.
            self.videoWriter = cv2.VideoWriter(self.output_video, 
                                    cv2.VideoWriter_fourcc(*'MJPG'),
                                    10, size)
            self.output_image_dir = "images"
            os.makedirs(self.output_image_dir,exist_ok=True)
        
    def init(self):
        self.cap = cv2.VideoCapture(self.video_filepath)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.detector = mp.tasks.vision.ObjectDetector.create_from_options(self.models.ObjecttDetectorOptions)
        print("FPS: ",self.fps)


        
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
    def plot(self,c_idx, bboxes, img):
        x1 = bboxes[0]
        y1 = bboxes[1]
        x2 = bboxes[2]
        y2 = bboxes[3]
        img = cv2.rectangle(img, (x1,y1-100),(x1+100,y1),color=(255,0,0),thickness=-1)
        img = cv2.putText(img, f'{c_idx}', org=(x1,y1),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=3,
                            color=(255,255,255),
                            thickness=4,
                            lineType=cv2.LINE_AA)
        img = cv2.rectangle(img, (x1,y1),(x2,y2),color=(0,255,0),thickness=5)
        return img
    
    # def run(self):
    #     frame_idx = 0
    #     # Embedding with tracking
    #     # Create Image Embedder
    #     crop_idx = 0

    #     all_detections=defaultdict(list)
    #     avg_processing_time = 0
    #     frame_interval = 1

    #     with vision.ImageEmbedder.create_from_options(self.models.emb_options) as embedder:
    #         with mp.tasks.vision.ObjectDetector.create_from_options(self.models.ObjecttDetectorOptions) as detector:
    #             while self.cap.isOpened():
    #                 ret, img = self.cap.read()
    #                 if ret == True:

    #                     frame_idx+=1
    #                     print("Processing frame : ",frame_idx)
    #                     # Calculate the timestamp of the current frame
    #                     frame_timestamp_ms = int(1000 * frame_idx / self.fps)
    #                     # print(frame_timestamp_ms)
    #                     time_start = time.perf_counter()
    #                     # if frame_idx%frame_interval==0:
    #                     # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    #                     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

    #                     # Perform object detection on the video frame.
    #                     detection_results = detector.detect_for_video(mp_image, frame_timestamp_ms)

    #                     obj_track = copy(self.obj_track)
    #                     for det_res in detection_results.detections:
    #                         x = det_res.bounding_box.origin_x
    #                         y = det_res.bounding_box.origin_y
    #                         width = det_res.bounding_box.width
    #                         height = det_res.bounding_box.height
                            
    #                         x1,y1,x2,y2 = (x,y,x+width,y+height)
    #                         # det_bboxes = (x1,y1,x2,y2)
    #                         # dets.append((x1,y1,x2,y2))

    #                         img_crop = img[y1:y2,x1:x2,:].astype(np.uint8)
    #                         crop_idx+=1
    #                         # Calculate the timestamp of the current frame
    #                         frame_timestamp_ms = int(1000 * crop_idx / self.fps)
    #                         # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    #                         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_crop)

    #                         # Perform object detection on the video frame.
    #                         embedding_result = embedder.embed_for_video(mp_image, frame_timestamp_ms)

    #                         # embed_arr.append(embedding_result.embeddings[0])
    #                         features = embedding_result.embeddings[0]
    #                         det_bbox = [x1,y1,x2,y2]
    #                         all_detections[frame_idx].append((det_bbox, features))
    #                         # img = cv2.rectangle(img, (x,y),(x+width,y+height),color=(0,255,0),thickness=2)
                        
    #                         if len(obj_track.keys())==0:

    #                             self.track_idx+=1

    #                             self.obj_track[self.track_idx] = self.obj_track.get(self.track_idx, Track(self.track_idx,bbox=det_bbox, features=features, frame_limit=self.frame_limit))
    #                             if self.visualize or self.save_output:
    #                                 img = self.plot(self.track_idx, det_bbox, img)

    #                         else:

    #                             em1 = features
    #                             best_match = 0
    #                             new_track_idx = None
    #                             # idx_remove = None
    #                             # best_iou = 0
    #                             # best_iou_idx = 0
    #                             # c_idx = 1
    #                             for c_idx, track in obj_track.items():
    #                                 # track = obj_track[c_idx]
    #                                 coords = track.get_bbox()
    #                                 em2 = track.features
    #                                 similarity = vision.ImageEmbedder.cosine_similarity(
    #                                     em1,
    #                                     em2)
                                    
    #                                 # iou = self.IOU(det_bbox, coords)

    #                                 if similarity>best_match:
    #                                     # print(similarity)
    #                                     best_match = similarity
    #                                     if similarity>self.feature_matching_thresh:
                                            
    #                                         new_track_idx = c_idx
    #                                         # idx_remove = idx
    #                             if new_track_idx is None:
    #                                 flag= True

                                            
    #                                 if flag:
    #                                     self.track_idx+=1
    #                                     new_track_idx = self.track_idx
    #                                     self.obj_track[self.track_idx] = self.obj_track.get(self.track_idx, Track(self.track_idx, bbox=det_bbox, features=features,frame_limit=self.frame_limit))
    #                                     pred_data = self.obj_track[new_track_idx].predict()
    #                                     yield (False,self.track_idx,pred_data)
    #                             else:
    #                                 x1,y1,x2,y2 = det_bbox
    #                                 pred_data = self.obj_track[new_track_idx].predict()
    #                                 # pred_bbox = self.obj_track[new_track_idx].get_bbox()
    #                                 self.obj_track[new_track_idx].update(bbox=det_bbox,features=features)
    #                                 del obj_track[new_track_idx]
    #                                 if self.visualize or self.save_output:
    #                                     img = self.plot(new_track_idx,det_bbox, img)
    #                                 yield (True, new_track_idx,pred_data)

                                
                        
    #                     # remove the tracking id if the limit is crossed
    #                     c_idxes = list(obj_track.keys())
    #                     while len(c_idxes)>0:
    #                         c_id = c_idxes.pop(0)
    #                         ret = self.obj_track[c_id].notFound()
    #                         if ret:
    #                             # cross the limit
    #                             del self.obj_track[c_id]

    #                     if self.save_output:
    #                         cv2.imwrite(os.path.join(self.output_image_dir,f"image_{frame_idx}.png"),img)
    #                         self.videoWriter.write(img)
    #                     if self.visualize:
    #                         cv2.imshow("Frame", img)
    #                         if cv2.waitKey(1) & 0xFF == ord('q'):
    #                             break

    #                     try:    
    #                         avg_processing_time+=(time.perf_counter()-time_start)/((frame_idx-1)//frame_interval)
    #                     except ZeroDivisionError as ZE:
    #                         avg_processing_time+=(time.perf_counter()-time_start)

    #                     time_processing = time.perf_counter()-time_start
    #                     print("time take to process frame : ", time_processing)
    #                     avg_processing_time += time_processing
    #                 else:
    #                     print("Not able to read the frame")
    #                     break

    #     self.cap.release()
    #     if self.save_output:
    #         self.videoWriter.release()
    #     print("time taken : ",avg_processing_time/frame_idx)


    def track_init(self):
        self.obj_track_temp = copy(self.obj_track)

    def track_close(self):
        # remove the tracking id if the limit is crossed
        c_idxes = list(self.obj_track_temp.keys())
        while len(c_idxes)>0:
            c_id = c_idxes.pop(0)
            ret = self.obj_track[c_id].notFound()
            if ret:
                # cross the limit
                del self.obj_track[c_id]

    def track(self,img, det_bbox):
    
        self.frame_idx+=1
        # for det_res in detection_results.detections:            
        
        x1,y1,x2,y2 = det_bbox
        img_crop = img[y1:y2,x1:x2,:].astype(np.uint8)
        # Calculate the timestamp of the current frame
        frame_timestamp_ms = int(1000 * self.frame_idx / self.fps)
        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_crop)

        # Perform object detection on the video frame.
        embedding_result = self.embedder.embed_for_video(mp_image, frame_timestamp_ms)

        # embed_arr.append(embedding_result.embeddings[0])
        features = embedding_result.embeddings[0]

        if len(self.obj_track.keys())==0:

            self.track_idx+=1

            self.obj_track[self.track_idx] = self.obj_track.get(self.track_idx, Track(self.track_idx,bbox=det_bbox, features=features, frame_limit=self.frame_limit))
            if self.visualize or self.save_output:
                img = self.plot(self.track_idx, det_bbox, img)
            
            pred_data = self.obj_track[self.track_idx].predict()
            return (False,self.track_idx,img,pred_data)

        else:

            em1 = features
            best_match = 0
            new_track_idx = None
            for c_idx, track in self.obj_track.items():
                # track = obj_track[c_idx]

                em2 = track.features
                similarity = vision.ImageEmbedder.cosine_similarity(
                    em1,
                    em2)
                
                # iou = self.IOU(det_bbox, coords)

                if similarity>best_match:
                    # print(similarity)
                    best_match = similarity
                    if similarity>self.feature_matching_thresh:
                        
                        new_track_idx = c_idx
                        # idx_remove = idx
            if new_track_idx is None:
                flag= True

                        
                if flag:
                    self.track_idx+=1
                    new_track_idx = self.track_idx
                    self.obj_track[self.track_idx] = self.obj_track.get(self.track_idx, Track(self.track_idx, bbox=det_bbox, features=features,frame_limit=self.frame_limit))
                    pred_data = self.obj_track[new_track_idx].predict()
                    return (False,self.track_idx,img,pred_data)
            else:
                x1,y1,x2,y2 = det_bbox
                pred_data = self.obj_track[new_track_idx].predict()
                # pred_bbox = self.obj_track[new_track_idx].get_bbox()
                self.obj_track[new_track_idx].update(bbox=det_bbox,features=features)
                if self.visualize or self.save_output:
                    img = self.plot(new_track_idx,det_bbox, img)
                return (True,new_track_idx,img,pred_data)


    def detect_track(self, idx, img):

        print("Processing frame : ",idx)
        # Calculate the timestamp of the current frame
        frame_timestamp_ms = int(1000 * idx / self.fps)
        # if frame_idx%frame_interval==0:
        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

        # Perform object detection on the video frame.
        detection_results = self.detector.detect_for_video(mp_image, frame_timestamp_ms)
        self.track_init()
        for det_res in detection_results.detections:
            x = det_res.bounding_box.origin_x
            y = det_res.bounding_box.origin_y
            width = det_res.bounding_box.width
            height = det_res.bounding_box.height
            
            x1,y1,x2,y2 = (x,y,x+width,y+height)
            det_bboxes = (x1,y1,x2,y2)

            flag, track_idx, img, pred_data = self.track(img,det_bboxes)                            
        self.track_close()
        return img 

    def video_run(self):
        idx = 0
        frame_interval = 1
        avg_processing_time = 0
        while self.cap.isOpened():
            ret, img = self.cap.read()
            idx+=1
            if ret == True:
                time_start = time.perf_counter()
                img = self.detect_track(idx, img)

                if self.save_output:
                    cv2.imwrite(os.path.join(self.output_image_dir,f"image_{idx}.png"),img)
                    self.videoWriter.write(img)
                if self.visualize:
                    cv2.imshow("Frame", img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                try:    
                    avg_processing_time+=(time.perf_counter()-time_start)/((idx-1)//frame_interval)
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
        print("time taken : ",avg_processing_time/idx)
             
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Detect and Sort tracking',
                    description='Detect and Sort tracking')
    
    parser.add_argument("-d", "--detect_model",type=str, default="model/object_detector/efficientdet_lite0_uint8.tflite")
    parser.add_argument("-e", "--embed_model",type=str, default="model/embedder/mobilenet_v3_small_075_224_embedder.tflite")
    parser.add_argument("--save_output_video","--sv",type=str,default='objEmbed_video_cars_track.avi')
    parser.add_argument("--frame_limit","--f", type=int, default=60, help="After this many consecutive frames the track id will be remove from the memory")
    parser.add_argument("--save",action = 'store_true',help = "save the video and images")
    parser.add_argument("--vis",action='store_true',help="visualise the tracking output for debug")
    args = parser.parse_args()

    deep_sort = Sort(
        video_filepath=0,
        detection_model_filepath=args.detect_model,
        embedding_model_filepath=args.embed_model,
        frame_limit=args.frame_limit,
        output_video=args.save_output_video,
        visualize=args.vis,
        save_output=args.save)
    # for bbox in deep_sort.run():
    #     print("bbox : ",bbox)
    deep_sort.init()
    deep_sort.video_run()