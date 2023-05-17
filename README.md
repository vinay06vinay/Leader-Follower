# Leader-Follower


##  Interactive files
* `interactive_aruco.ipynb` contains the aruco marker code.
* `interactive_DO.ipynb` contains the code for object detection and embedding model with kalman filter tracking.

## python files
* `sort.py` contains tracking algorithm, using embedding between 2 frames and the if the frames embedding match the track id moves to the frame, else new track id is assigned.

* `sort_v2.py` updated version of the `sort.py` where the tracking is independent of 2 consecutive frames, so more robust and can track the object even after occlusion.
    - Get the argument details:
        `python3 sort_v2.py -h` 

## run the tracking script
> python3 sort_v2.py

* Note: assuming the dataset and model are placed in the following directory format.
    - model/
        - embedder/
            - mobilenet_v3_large.tflite
        - object_detector/
            - efficientdet_lite0_uint8.tflite
    - dataset/
        - video_cars.mp4

## requirements
> pip install -r requirements.txt

## dataset
* [video_cars.mp4](https://drive.google.com/file/d/119pDDZhH64BOW-6NPdEfDu3s_8-OA_4A/view?usp=sharing)

## models
* embedding model, choose according to requirement:
    - [embedding_model](https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/latest/mobilenet_v3_small.tflite)

    - [embedding_model_large](https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_large/float32/latest/mobilenet_v3_large.tflite)

* [object_detection_model](https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite)


## References
* [Image Embedding models](https://developers.google.com/mediapipe/solutions/vision/image_embedder/index#models)
* [Image Embedding Mediapipe](https://developers.google.com/mediapipe/solutions/vision/image_embedder/python)
* [Object Detector models](https://developers.google.com/mediapipe/solutions/vision/object_detector#efficientdet-lite0_model_recommended)
* [Object Detector Mediapipe](https://developers.google.com/mediapipe/solutions/vision/object_detector/python)
* [Aruco Marker OpenCV](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
* [midas depth colab](https://colab.research.google.com/drive/1QjcqchMme2gFqoaLsAcg0eNa7s-xfN8W#scrollTo=expanded-verification)
* 
## How to add this repo to other repo
* https://dev.to/jjokah/submodules-a-git-repo-inside-a-git-repo-36l9
