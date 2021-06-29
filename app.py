import streamlit as st
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image

import tempfile
st.title("Object Detection")
st.sidebar.markdown("# Model")
confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)


net = cv2.dnn.readNetFromDarknet('yolov3-tiny_obj.cfg', 'yolov3-tiny_obj_best.weights')
classes = ['headphone', 'wallet']
layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size=(len(classes),3))
run=st.checkbox('Open/Close your Webcam')
video = cv2.VideoCapture(0)

 
starting_time = time.time()
frame_id = 0
font = cv2.FONT_HERSHEY_PLAIN

no_of_classes = []
confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
frame_image = st.empty()
def welcome ():
  
    
    st.title('Image Processing using Streamlit')
    
    st.subheader('A simple app that shows different object detections from different formats. You can choose the options'
             + ' from the left.')

			 def photo():
    #confidence_threshold = st.sidebar.slider("Confidence threshold", 0, 100, DEFAULT_CONFIDENCE_THRESHOLD, 1)
    uploaded_file = st.sidebar.file_uploader(" ", type=['png', 'jpg', 'jpeg'])
        
    
    def detection(frame):
        #frame = np.array(frame)
        height,width = frame.shape[:2]
        #height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB =True, crop = False)
        net.setInput(blob)
        outs = net.forward(outputlayers)
        boxes = []
        confidences = []
        class_ids = []
        for output in outs:
            for detection in output:
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                if confidence > .6:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    boxes.append([x,y,w,h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold,.4)
        a=""
		for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x,y), (x+w, y+h), color,2)
                a = a+" "+ label
                cv2.putText(frame, label + ' ' + str(round(confidence,2)), (x,y+30), font, 5, (255,255,255),2)
        st.image(frame)
        st.write("Detected Object: " +a)
    if uploaded_file:
        frame = Image.open(uploaded_file)
        frame = np.array(frame)
        detection(frame)
		
def video():
    f = st.sidebar.file_uploader("Upload file")
    starting_time = time.time()
    frame_id = 0
    frame_window = st.image([])
    frame_text = st.markdown("")
    frame_text1 = st.markdown("")
    frame_text2 = st.empty()
    frame_text3 = st.markdown("")
    frame_text4 = st.empty()
    if f:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(f.read())
        vf = cv2.VideoCapture(tfile.name)
    

        while vf.isOpened():
            _,frame=vf.read()
			
			
			height, width, channels = frame.shape
            # detecting objects
            blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0,0,0), swapRB =True, crop = False)
            net.setInput(blob)
            outs = net.forward(outputlayers)
            boxes = []
            confidences = []
            class_ids = []
            for output in outs:
                for detection in output:
                    score = detection[5:]
                    class_id = np.argmax(score)
                    confidence = score[class_id]
                    if confidence > .6:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2]*width)
                        h = int(detection[3]*height)
                        x = int(center_x - w/2)
                        y = int(center_y - h/2)
                        boxes.append([x,y,w,h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold,.4)
            t=""
            t_ = ""
            n=0
            n_ = 0
            for i in range(len(boxes)):
                if i in indexes:
                    x,y,w,h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame, (x,y), (x+w, y+h), color,2)
                    a = str(round(confidence,2))
					
					 cv2.putText(frame, label + ' ' + str(round(confidence,2)), (x,y+30), font, 1, (255,255,255),2)
                    t = t + " " + label
                    n+= 1
                    
                    #frame_text4.text("Items missing : "+)
                else:

                
                    label_=str(classes[~class_ids[i]])
                    t_ = t_ + " " + label_
                    #frame_text2.text("Items missing : "+t_)
                    
                #frame_text2.text("Items missing : "+t_)
                    
					
			elapsed_time = time.time() - starting_time
            fps=frame_id/elapsed_time
            #code to change the blue intensive video to normal
            img_np = np.array(frame)
            frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            #adding the frame to above created empty frame_window
            frame1 = cv2.putText(frame, 'FPS:'+str(round(fps,2)), (10,50), font, 2, (0,0,0),1)
            frame_window.image(frame1)
            frame_text3.text("Items Found : "+t)
            frame_text4.text("No. of Items Found: "+str(n))
			
			def object_detection():
    run=st.checkbox('Open/Close your Webcam')
    video = cv2.VideoCapture(0)
    starting_time = time.time()
    frame_id = 0
    frame_window = st.image([])
    frame_text = st.markdown("")
    frame_text1 = st.markdown("")
    frame_text2 = st.empty()
    frame_text3 = st.markdown("")
    frame_text4 = st.empty()
    while run:
        _,frame=video.read()
        #a = frame_window.image(frame)
        #st.write(a)

        height, width, channels = frame.shape
        # detecting objects
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0,0,0), swapRB =True, crop = False)
        net.setInput(blob)
        outs = net.forward(outputlayers)
        boxes = []
        confidences = []
        class_ids = []
        for output in outs:
            for detection in output:
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                if confidence > .6:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    boxes.append([x,y,w,h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold,.4)
        t=""
        t_ = ""
        n=0
        n_ = 0
        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x,y), (x+w, y+h), color,2)
                a = str(round(confidence,2))
                cv2.putText(frame, label + ' ' + str(round(confidence,2)), (x,y+30), font, 1, (255,255,255),2)
                t = t + " " + label
                n+= 1
                
                #frame_text4.text("Items missing : "+)
            else:

            
                label_=str(classes[~class_ids[i]])
                t_ = t_ + " " + label_
                #frame_text2.text("Items missing : "+t_)
                
            #frame_text2.text("Items missing : "+t_)
                
                
                

            
        elapsed_time = time.time() - starting_time
        fps=frame_id/elapsed_time
        #code to change the blue intensive video to normal
        img_np = np.array(frame)
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        #adding the frame to above created empty frame_window
        frame1 = cv2.putText(frame, 'FPS:'+str(round(fps,2)), (10,50), font, 2, (0,0,0),1)
        frame_window.image(frame1)
        frame_text3.text("Items Found : "+t)
        frame_text4.text("No. of Items Found: "+str(n))

selected_box = st.sidebar.selectbox("Select mode",
                       [
                        "Welcome",
                       "Upload a photo",
                       "Upload a video",
                       "Open up camera"] )
if selected_box == "Welcome":
    welcome ()
if selected_box == "Upload a photo":

    photo()
if selected_box == "Upload a video":
    video()
if selected_box == "Open up camera":
    object_detection()





