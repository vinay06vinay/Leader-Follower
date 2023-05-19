# ENPM 673 - Project 5

## Team Members
* Abhijay Singh (UID - 118592619)
* Tharun Puthanveettil (UID - 119069516)
* Yashveer Jain (UID - 119252864)
* Sameer Arjun S (UID - 119385876)
* Vinay Bukka (UID - 18176680)

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

## Execution

* **Leader-Follower** 
    
    - To execute the program, first change the IP address on line 31 in the file `main.py` to the IP address of your computer.
    - To execute the program for this problem, navigate to the submission folder and use the following commands
        ```
        python3 main.py
        ```

