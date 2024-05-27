# ENPM 673 - Project 5

## Team Members
* Abhijay Singh (UID - 118592619)
* Tharun Puthanveettil (UID - 119069516)
* Yashveer Jain (UID - 119252864)
* Sameer Arjun S (UID - 119385876)
* Vinay Bukka (UID - 18176680)

## Citation and Paper 
If you find Leader Follower useful in your academic work, please cite the [paper](https://arxiv.org/pdf/2405.11659).

## Setup

- Create a python virtual environment to install the dependencies for the project. 
    ```
    python3 -m venv <path_to_virtual_env>
    source <path_to_virtual_env>/bin/activate
    ```

- Install the dependencies using the following command:
    ```
    pip3 install -r requirements.txt
    ```

- Download the weights for the following models using the provided links:
    - [MiDaS](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt)
    - [Embedder](https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_large/float32/latest/mobilenet_v3_large.tflite)
    - [Custom YOLOv7](https://drive.google.com/file/d/12Xgb0qlIBJmL-IOISdWeSAcuhFNiTbKv/view?usp=share_link)

- Create a folder named weights and place the downloaded weights in the folder.

---

## Execution

* **Leader-Follower** 
    
    - To execute the program, first change the IP address on line 31 in the file `main.py` to the IP address of your computer.
    - To execute the program for this problem, navigate to the submission folder and use the following commands
        ```
        python3 main.py
        ```
---

## Details
* `sort.py` is referred from `sort_v2.py` in tracking branch.

### Instructions To Run The Follower Code
1. Once the main server is started, follower.py code is run.
2. To Run "follower.py" code in Rasberry Pi, first you need to ssh into Rasberry pi of the follower robot using "ssh username@password". 
3. Transfer the script file to the Rasberry Pi and install all necessary libraries such as opencv, flask, socket, imutils, numpy, gpio.
4. Open a terminal and run the "follower.py" file while the main server of socket is running.

### Instructions To Run The Leader Code
1. Once the main server is started, leader.py code is run.
2. To Run "leader.py" code in Rasberry Pi, first you need to ssh into Rasberry pi of the Leader robot using "ssh username@password". 
3. Transfer the script file to the Rasberry Pi and install all necessary libraries such as opencv, flask, socket, imutils, numpy, gpio.
4. Open a terminal and run the "leader.py" file while the main server of socket is running.

### Videos
* Results: The follower will be following the leader with a threshold distance as long as it tracks the leader. If a Dynamic Obstacle comes into the play the follower pauses the motion and communicates with the leader and other followers to stop. Once the obstacle goes out of the frame and follower is tracking the leader the whole platoon system starts moving again. [Link](https://drive.google.com/drive/u/1/folders/1lEYSDns3Q3QMbOsjFyqpdVxPJQgv3-mY?authuser=0)
