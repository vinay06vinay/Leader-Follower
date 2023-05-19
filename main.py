import socket
import struct
import pickle
import argparse
import time
import numpy as np
import random

import cv2
import torch
import torch.backends.cudnn as cudnn

# YOLO
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression,  scale_coords, strip_optimizer, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

# MiDaS
from midas.model_loader import load_model

from sort import Sort

from flask import Flask, render_template, request,  jsonify
import threading

HOST='10.125.129.5'  #  CHANGE THIS TO YOUR IP ADDRESS



#Dictionary to store the tracking data and the depth data
tracking_data = {}
depth_data = {}


# Boolean variable to detect the tracked object
tracker_detected = False
first_execution = True


follower1_status = "active"
follower2_status = "active"
follower1_command = "stop"
follower2_command = "stop"
system_status = {'follower1_status': follower1_status, 'follower2_status': follower2_status}
system_command = {'follower1_command': follower1_command, 'follower2_command': follower2_command}

# Dictionary to store the system status, tracker_data, depth_data
perception_output = {'system_status': system_status, 'tracker_data': tracking_data, 'depth_data': depth_data}


app = Flask(__name__)
@app.route('/')
def student():
   return render_template('student.html')


@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      return render_template("result.html",result = result)


@app.route('/follower1_status_update', methods=['POST'])
def follower1_status_update():
   global system_status
   #global tracker_detected
   status = request.form.get('status')
   print("Follower 1 reported: " + status)
#    if status == "obstacle":
#       follower1_status = "deactive"
#       follower2_status = "deactive"
#       follower1_command = "stop"
#       follower2_command = "stop"
#       return follower1_command
#    else:
#       follower1_status = "active"
#       follower2_status = "active"
#       follower1_command = "start"
#       follower2_command = "start"
#       return follower1_command


@app.route('/follower2_status_update', methods=['POST'])
def follower2_status_update():
   global system_status
   global system_command
   status = request.form.get('status')
   print("Follower 2 reported: " + status)
   if status == "obstacle":
      system_status['follower1_status'] = "deactive"
      system_status['follower2_status'] = "deactive"
      system_command['follower1_command'] = "stop"
      system_command['follower2_command'] = "stop"
      return follower2_command
   
@app.route('/system_status', methods=['GET'])
def report_system_status():
    global system_status
    global tracker_detected
    global tracking_data
    global depth_data
    global tracking_variables
    #Store the tracker id, x,y and 
    tracking_variables['system_status'] = system_status
    tracking_variables['tracker_detected'] = tracker_detected
    tracking_variables['tracking_data'] = tracking_data
    tracking_variables['depth_data'] = depth_data


    # print("Tracker detected: " + str(tracker_detected))
    if tracker_detected == True:
        system_status['follower1_status'] = "active"
        system_status['follower2_status'] = "active"
        system_command['follower1_command'] = "start"
        system_command['follower2_command'] = "start"
    elif tracker_detected == False:
        system_status['follower1_status'] = "deactive"
        system_status['follower2_status'] = "deactive"
        system_command['follower1_command'] = "stop"
        system_command['follower2_command'] = "stop"
    # print(system_status)

    return jsonify(tracking_variables)

@app.route('/system_command', methods=['GET'])
def report_system_command():
   global system_command
   return jsonify(system_command)


@app.route('/check_tracker_status', methods=['GET'])
def check_tracker_status():
   global tracker_detected 
   print("Sysstem status: " + system_status['follower1_status'] + " " + system_status['follower2_status'])
   if tracker_detected == True:
        status_string = "Tracker Detected"
   else:
        status_string = "Tracker Not Detected"
   render_template('index.html', message=status_string)



def process(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    global first_execution

    sample = torch.from_numpy(image).to(device).unsqueeze(0)

    if optimize and device == torch.device("cuda"):
        if first_execution:
            print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                      "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                      "  half-floats.")
        sample = sample.to(memory_format=torch.channels_last)
        sample = sample.half()

    if first_execution or not use_camera:
        height, width = sample.shape[2:]
        print(f"    Input resized to {width}x{height} before entering the encoder")
        first_execution = False

    prediction = model.forward(sample)
    prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return prediction

def create_side_by_side(image, depth, grayscale):
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)
    normalized_depth *= 3

    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
    if not grayscale:
        right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)

    if image is None:
        return right_side
    else:
        return np.concatenate((image, right_side), axis=1)

def run( grayscale=True):
    global tracker_detected
    global tracking_data
    global depth_data

    weights, imgsz = opt.weights, opt.img_size
    model_path, model_type, = "./weights/dpt_swin2_tiny_256.pt", "dpt_swin2_tiny_256"

    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  

    yolo_model = attempt_load(weights, map_location=device)  
    stride = int(yolo_model.stride.max())  
    imgsz = check_img_size(imgsz, s=stride)  
    if half:
        yolo_model.half()  

    cudnn.benchmark = True 
    names = yolo_model.module.names if hasattr(yolo_model, 'module') else yolo_model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if device.type != 'cpu':
        yolo_model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(yolo_model.parameters())))  # run once

    print("Initialize MiDaS")
    midas_model, transform, net_w, net_h = load_model(device, model_path, model_type, False, None, False)

    sort = Sort(detection_model_filepath="weights/efficientdet_lite0.tflite",
        embedding_model_filepath="weights/mobilenet_v3_large.tflite")

    # Socket for receiving frames
    
    PORT=65000
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')
    s.bind((HOST,PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')
    conn, addr = s.accept()
    data = b""
    payload_size = struct.calcsize(">L")
    print("payload_size: {}".format(payload_size))
    print("Start processing")

    with torch.no_grad():
        fps = 1
        time_start = time.time()
        frame_index = 0
        while True:
            while len(data) < payload_size:
                # print("Recv: {}".format(len(data)))
                data += conn.recv(4096)

            # print("Done Recv: {}".format(len(data)))
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            # print("msg_size: {}".format(msg_size))
            while len(data) < msg_size:
                data += conn.recv(4096)
            frame_data = data[:msg_size]
            data = data[msg_size:]

            frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

            "YOLOv7 Detection"
            if frame is not None:
                img = frame.copy()
                img = letterbox(img, imgsz, stride=stride)[0]
                img = img[:, :, ::-1].transpose(2, 0, 1)  
                img = np.ascontiguousarray(img)

                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                t1 = time_synchronized()
                with torch.no_grad():   
                    pred = yolo_model(img, augment=opt.augment)[0]
                t2 = time_synchronized()

                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                t3 = time_synchronized()

                sort.track_init()
                pred_data = None
                for i, det in enumerate(pred):  
                    im0 = frame.copy()
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                        
                        for *xyxy, conf, cls in reversed(det):
                            xyxy = list(map(lambda x: int(x), xyxy))
                            flag, track_idx, img_, pred_data = sort.track(im0, xyxy)
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                            if flag:
                                tracker_detected = True
                                # Store tracker output into tracker_data
                                tracking_data["track_idx"] = track_idx
                                tracking_data["img"] = img_
                                tracking_data["pred_data"] = pred_data
                            else:
                                tracker_detected = False
                    else:
                        tracker_detected = False

                sort.track_close()
                cv2.imshow('YOLOv7 Detection - Press Escape to close window ', im0)
                cv2.waitKey(1)

            "MiDaS Depth Estimation"
            if frame is not None and frame_index % 15 == 0:
                original_image_rgb = np.flip(frame, 2)
                image = transform({"image": original_image_rgb/255})["image"]

                prediction = process(device, midas_model, model_type, image, (net_w, net_h),
                                         original_image_rgb.shape[1::-1], False, True)

                depth = (5079.882 * 40)/ prediction
                depth = np.clip(depth, 0, 1000)
                if pred_data is not None:
                    dd = depth[int(pred_data[0]), int(pred_data[1])]
                    print("Depth at which the person is : ", dd)
                    depth_data["depth_agg"] = dd

                original_image_bgr = np.flip(original_image_rgb, 2) if False else None
                content = create_side_by_side(original_image_bgr, prediction, grayscale)
                cv2.imshow('MiDaS Depth Estimation - Press Escape to close window ', content/255)

                alpha = 0.1
                if time.time()-time_start > 0:
                    fps = (1 - alpha) * fps + alpha * 1 / (time.time()-time_start) 
                    time_start = time.time()
                
                print(f"\rFPS: {round(fps,2)}", end="")

            if cv2.waitKey(1) == 27:  # Escape key
                break

            frame_index += 1

            if frame is not None:
                print(f"\rProcessed {frame_index} frames", end="")

    print("Finished")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    port = 5000   
    t_webApp = threading.Thread(target=lambda: app.run(host=HOST, port=port, debug=True, use_reloader=False))
    t_webApp.setDaemon(True)
    t_webApp.start()

    run()