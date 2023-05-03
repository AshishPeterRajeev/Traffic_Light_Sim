import math
import cv2
import time
import os
from threading import Thread
import numpy as np
import imutils
import time
from scipy import spatial
import cv2

list_of_vehicles = ["bicycle", "car", "motorbike", "bus", "truck", "train"]
# camera_feeds = {"lane1": 0,"lane2": 1,"lane3": 2,"lane4": 3,}
camera_feeds = {"Lane 1": "traffic1.mp4","Lane 2": "traffic2.mp4","Lane 3": "traffic3.mp4","Lane 4": "traffic4.mp4",}

green_times = {"Lane 1": 10,"Lane 2": 15,"Lane 3": 18,"Lane 4": 20,}

yellow_time = 5

class_counts = {}
FRAMES_BEFORE_CURRENT = 10
inputWidth, inputHeight = 416, 416

defaultMinimum = 10
defaultMaximum = 60

carTime = 2
bikeTime = 1
rickshawTime = 2.25 
busTime = 2.5
truckTime = 2.5

labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])

inputVideoPath = "traffic1.mp4"
preDefinedConfidence = 0.5
preDefinedThreshold = 0.3

USE_GPU = 0


np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

class CustomThread(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
 
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
             
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

def boxInPreviousFrames(previous_frame_detections, current_box, current_detections):
    centerX, centerY, width, height = current_box
    dist = np.inf

    for i in range(FRAMES_BEFORE_CURRENT):
        coordinate_list = list(previous_frame_detections[i].keys())
        if len(coordinate_list) == 0:
            continue

        temp_dist, index = spatial.KDTree(
            coordinate_list).query([(centerX, centerY)])
        if (temp_dist < dist):
            dist = temp_dist
            frame_num = i
            coord = coordinate_list[index[0]]

    if (dist > (max(width, height)/2)):
        return False

    current_detections[(centerX, centerY)
                       ] = previous_frame_detections[frame_num][coord]
    return True

def count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame, list_of_vehicles):
    current_detections = {}
    

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            centerX = x + (w//2)
            centerY = y + (h//2)

            class_id = classIDs[i]

            if LABELS[class_id] in list_of_vehicles:
                vehicle_type = LABELS[class_id]


                if vehicle_type not in class_counts:
                    class_counts[vehicle_type] = 0

                current_detections[(centerX, centerY)] = (
                    vehicle_count, vehicle_type)
                

                if not boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections):
                    vehicle_count += 1
                    class_counts[vehicle_type] += 1

                vehicle_id, vehicle_type = current_detections.get(
                    (centerX, centerY))

                if list(current_detections.values()).count((vehicle_id, vehicle_type)) > 1:
                    current_detections[(centerX, centerY)] = (
                        vehicle_count, vehicle_type)
                    vehicle_count += 1
                    class_counts[vehicle_type] += 1

                cv2.putText(frame, str(vehicle_id), (centerX, centerY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)

    return vehicle_count, current_detections, class_counts




def get_vehicle_count(camera_feed):
    car_count,bus_count,motorbike_count,bicycle_count,truck_count,vehicle_count=0,0,0,0,0,0
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


    if USE_GPU:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    ln = net.getLayerNames()

    # ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    # ln = [ln[i - 1] for i in map(int, net.getUnconnectedOutLayers())]
    ln = [int(i) - 1 for i in net.getUnconnectedOutLayers()]




    videoStream = cv2.VideoCapture(camera_feed)
    # videoStream = cv2.VideoCapture(inputVideoPath)
    video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))


    x1_line = 0
    y1_line = video_height//2
    x2_line = video_width
    y2_line = video_height//2


    previous_frame_detections = [{(0, 0): 0} for i in range(FRAMES_BEFORE_CURRENT)]

    num_frames, vehicle_count = 0, 0

    start_time = int(time.time())

    while True:
        
        num_frames += 1
        

        boxes, confidences, classIDs = [], [], []
        vehicle_crossed_line_flag = False

    

        (grabbed, frame) = videoStream.read()

        if not grabbed:
            break

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inputWidth, inputHeight),
                                 swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        for output in layerOutputs:

            for i, detection in enumerate(output):

                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > preDefinedConfidence:

                    box = detection[0:4] * \
                        np.array([video_width, video_height,
                             video_width, video_height])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, preDefinedConfidence,
                            preDefinedThreshold)

        vehicle_count, current_detections, class_counts = count_vehicles(
        idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame, list_of_vehicles)

    
        t1 = int(time.time())
        cap_timer = t1-start_time
        # cv2.imshow('Frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
        if cv2.waitKey(1) & cap_timer>4 :
            break
    # 0xFF == ord('q')
    # if cap_timer>10:
    # 	break
    
        previous_frame_detections.pop(0)

        previous_frame_detections.append(current_detections)

    # print("[INFO] cleaning up...")

    videoStream.release()


    if "bicycle" in class_counts:
        bicycle_count = class_counts['bicycle']
        class_counts['bicycle']=0
    else:
        bicycle_count=0

    if "car" in class_counts:
        car_count = class_counts['car']
        class_counts['car']=0
    else:
        bicycle_count=0

    if "motorbike" in class_counts:
        motorbike_count = class_counts['motorbike']
        class_counts['motorbike']=0
    else:
        motorbike_count=0
    if "bus" in class_counts:
        bus_count = class_counts['bus']
        class_counts['bus']=0
    else:
        bus_count=0

    if "truck" in class_counts:
        truck_count = class_counts['truck']
        class_counts['truck']=0
    else:
        truck_count=0
    # class_counts = {}
    # print(car_count,bus_count,motorbike_count,bicycle_count,truck_count,vehicle_count)
    
    vehicle_counts = [0,0,0,0,0,0]
    vehicle_counts[0] = car_count
    vehicle_counts[1] = bus_count
    vehicle_counts[2] = motorbike_count
    vehicle_counts[3] = bicycle_count
    vehicle_counts[4] = truck_count
    vehicle_counts[5] = vehicle_count
    # print(vehicle_counts)
    # return car_count,bus_count,motorbike_count,bicycle_count,truck_count,vehicle_count
    return vehicle_counts




def adjust_green_time(vehicle_count):
    # replace this with your code to adjust the green time based on the vehicle count
    # and return the new green time
    car_count,bus_count,motorbike_count,bicycle_count,truck_count,vehicle_count=vehicle_count
    greenTime = math.ceil((car_count*carTime)  + (bus_count*busTime) + (truck_count*truckTime)+ (motorbike_count*bikeTime))
    greenTime = max(defaultMinimum, min(defaultMaximum, greenTime))
    
    # print("Count",car_count,bus_count,motorbike_count,bicycle_count,truck_count,vehicle_count)
    return greenTime

def signal(lane,sigTime,flag):
    # capture_object = cap
    if(flag==1):
        while sigTime>0:
            sigTime=sigTime-1
            print(lane, "-> GREEN [",sigTime+1,"]")
            time.sleep(1)
            # if(sigTime==5):
                # vehicle_count_thread = ThreadWithReturnValue(target=get_vehicle_count, args=(capture_object,))
                # vehicle_count_thread.start()
    elif(flag==2):
        while sigTime>0:
            sigTime=sigTime-1
            print(lane, "-> YELLOW [",sigTime+1,"]")
            time.sleep(1)

def run_traffic_light_system(lane):
   
    while True:
        greentime = green_times[lane]
        next_cam_lane = "Lane " + str(int(lane[5])+1)
        if next_cam_lane == "Lane 5":
            next_cam_lane = "Lane 1"
        capture_object = camera_feeds[next_cam_lane]
        # print(capture_object)

        vehicle_count_thread = CustomThread(target=get_vehicle_count, args=(capture_object,))
        
        signal(lane,greentime,1)
        vehicle_count_thread.start()
        
        print("----------------------------------------")
        signal(lane,yellow_time,2)
        
        vehicle_count = vehicle_count_thread.join()
        # print("Vehicle",vehicle_count)
        
        next_lane = "Lane " + str(int(lane[5])+1)
        if next_lane == "Lane 5":
            next_lane = "Lane 1"
        new_greentime = adjust_green_time(vehicle_count)
        # greentime = green_times[next_lane] if new_greentime > 0 else greentime
        green_times[next_lane] = new_greentime
        # turn on the red light and move to the next lane
        print("----------------------------------------")
        print(lane, "-> RED") 
        print("----------------------------------------")
        print("Next lane :", next_lane, "|| Green Time ->", new_greentime, "seconds.")
        print("----------------------------------------")
        time.sleep(1)
                
        lane = next_lane

# start the traffic light system for each lane
# for lane in camera_feeds.keys():
#     run_traffic_light_system(lane)
run_traffic_light_system("Lane 1")