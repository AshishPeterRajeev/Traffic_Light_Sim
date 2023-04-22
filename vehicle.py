import numpy as np
import imutils
import time
from scipy import spatial
import cv2
import os
# from input_retrieval import *


list_of_vehicles = ["bicycle", "car", "motorbike", "bus", "truck", "train"]
# vehicle_type=
class_counts = {}
FRAMES_BEFORE_CURRENT = 10
inputWidth, inputHeight = 416, 416
vehicle_count=0
bicycle_count = 0
car_count = 0
motorbike_count = 0
bus_count = 0
truck_count = 0


# LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath,\
# 	preDefinedConfidence, preDefinedThreshold, USE_GPU= parseCommandLineArguments()
# print(type(preDefinedConfidence))

labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])

# inputVideoPath = input("Enter the video input name: ")
# preDefinedConfidence = float(input("Enter confidence score: "))
# preDefinedThreshold = float(input("Enter threshold value: "))

inputVideoPath = "traffic4.mp4"
preDefinedConfidence = 0.5
preDefinedThreshold = 0.3

USE_GPU = 1


np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")


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

# def count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame):
# 	current_detections = {}

# 	if len(idxs) > 0:

# 		for i in idxs.flatten():

# 			(x, y) = (boxes[i][0], boxes[i][1])
# 			(w, h) = (boxes[i][2], boxes[i][3])

# 			centerX = x + (w//2)
# 			centerY = y+ (h//2)

# 			if (LABELS[classIDs[i]] in list_of_vehicles):
# 				current_detections[(centerX, centerY)] = vehicle_count
# 				if (not boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections)):
# 					vehicle_count += 1

# 				ID = current_detections.get((centerX, centerY))


# 				if (list(current_detections.values()).count(ID) > 1):
# 					current_detections[(centerX, centerY)] = vehicle_count
# 					vehicle_count += 1


# 				cv2.putText(frame, str(ID), (centerX, centerY),\
# 					cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 2)

# 	return vehicle_count, current_detections




def count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame, list_of_vehicles):
    current_detections = {}
    # class_counts = {}

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            centerX = x + (w//2)
            centerY = y + (h//2)

            class_id = classIDs[i]

            if LABELS[class_id] in list_of_vehicles:
                vehicle_type = LABELS[class_id]
                # print(vehicle_type)
                # class_counts[]

                if vehicle_type not in class_counts:
                    class_counts[vehicle_type] = 0

                current_detections[(centerX, centerY)] = (
                    vehicle_count, vehicle_type)
                # print(class_counts)

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







print("[INFO] loading YOLO from disk...")


def vstart():
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


    if USE_GPU:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    ln = net.getLayerNames()

    # ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    ln = [int(i) - 1 for i in net.getUnconnectedOutLayers()]
    


    videoStream = cv2.VideoCapture(inputVideoPath)
    video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))


    x1_line = 0
    y1_line = video_height//2
    x2_line = video_width
    y2_line = video_height//2


    previous_frame_detections = [{(0, 0): 0} for i in range(FRAMES_BEFORE_CURRENT)]

    num_frames, vehicle_count = 0, 0
# writer = initializeVideoWriter(video_width, video_height, videoStream)
    start_time = int(time.time())

    while True:
        #print("================NEW FRAME================")
        num_frames += 1
        #print("FRAME:\t", num_frames)

        boxes, confidences, classIDs = [], [], []
        vehicle_crossed_line_flag = False

    # start_time, num_frames = displayFPS(start_time, num_frames)

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

    # drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)

    # vehicle_count, current_detections, class_counts = count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame, list_of_vehicles)

    # displayVehicleCount(frame, vehicle_count)

    # writer.write(frame)
        t1 = int(time.time())
        cap_timer = t1-start_time
        cv2.imshow('Frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
        if cv2.waitKey(1) & cap_timer>10 :
            break
    # 0xFF == ord('q')
    # if cap_timer>10:
    # 	break
    
        previous_frame_detections.pop(0)

        previous_frame_detections.append(current_detections)

    print("[INFO] cleaning up...")
    #print("Vehicles:", vehicle_count)
    #print(class_counts)
# writer.release()
    videoStream.release()


    if "bicycle" in class_counts:
        bicycle_count = class_counts['bicycle']
    else:
        bicycle_count=0

    if "car" in class_counts:
        car_count = class_counts['car']
    else:
        bicycle_count=0

    if "motorbike" in class_counts:
        motorbike_count = class_counts['motorbike']
    else:
        motorbike_count=0
    if "bus" in class_counts:
        bus_count = class_counts['bus']
    else:
        bus_count=0

    if "truck" in class_counts:
        truck_count = class_counts['truck']
    else:
        truck_count=0
    print(car_count,bus_count,motorbike_count,bicycle_count,truck_count,vehicle_count)
    return car_count,bus_count,motorbike_count,bicycle_count,truck_count,vehicle_count

vstart()