"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

# Setup log file
log.basicConfig(filename='logfile.log', filemode='w', level=log.DEBUG)

# Get correct video codec
log.info("Get correct video codec...")
if sys.platform == "linux" or sys.platform == "linux2":
    CODEC = 0x00000021
elif sys.platform == "darwin":
    CODEC = cv2.VideoWriter_fourcc('M','J','P','G')
else:
    log.error("Unsupported OS.")
    exit(1)

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def parse_output(frame, result, prob_threshold, w, h):
    '''
    Parse output.
    '''

    count = 0
    for obj in result[0][0]:
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * w)
            ymin = int(obj[4] * h)
            xmax = int(obj[5] * w)
            ymax = int(obj[6] * h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            count += 1
    return frame, count

def print_out_conf(result, frame):
    '''
    Print mean output confidence of people detected on frame.
    '''
    count = 0
    tot_confidence = 0
    mean_out_conf = 0
    
    for obj in result[0][0]:
        if obj[2] > 0:
            tot_confidence += obj[2]
            count += 1
    
    if count > 0:
        mean_out_conf = tot_confidence / count
        
    message = "Output confidence: {:.1f}%".format(mean_out_conf * 100)
    cv2.putText(frame, message, (20, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return

def print_inf_time(time, frame):
    '''
    Print inference time on frame.
    '''
    message = "Inference time: {:.1f}ms".format(time * 1000)
    cv2.putText(frame, message, (20, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return

def connect_mqtt():
    '''
    Connect to the MQTT client
    '''
    log.info("Connect to the MQTT client...")
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    
    # Initialise counters
    current_count = 0
    last_count = 0
    debounced_count = 0
    debounced_last_count = 0
    total_count = 0

    # Load the model through `infer_network`
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()

    # Handle webcam, image or video  
    log.info("Opening input stream...")
    # Create a flag for single images
    image_flag = False
    # Check input
    if args.input == 'CAM':
        # The input is a webcam
        log.info("Webcam input.")
        input_stream = 0
    elif os.path.isfile(args.input) == True:
        if args.input.endswith('.jpg') or args.input.endswith('.bmp'):
            # The input is a single image
            log.info("Single image input.")
            image_flag = True
        else:
            # The input is a video
            log.info("Video input.")
        input_stream = args.input
    else:
        # The input is not valid
        log.error("Specified input doesn't exist.")
        exit(1)      
        
    # Get and open video capture
    try:
        cap = cv2.VideoCapture(input_stream)
        cap.open(input_stream)
        log.info("Input correctly opened.")
    except:
        log.error("Unable to open input.")
        exit(1)
            
    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Initialize times
    start_ptime = 0.0
    start_debounce_time = 0.0
    
    log.info("Processing frames...")
    
    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        # Pre-process the frame
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        # Inference start time
        inf_start_t = time.time()
        
        # Perform inference on the frame
        infer_network.exec_net(0, p_frame)

        # Wait for the result
        if infer_network.wait(0) == 0:
            #Compute inference time
            inf_t = time.time() - inf_start_t
            
            # Get the results of the inference request
            result = infer_network.get_output(0)

            # Extract any desired stats from the results
            out_frame, current_count = parse_output(frame, result, args.prob_threshold, width, height)
            
            # "Debounce" mechanism
            if current_count != last_count:
                start_debounce_time = time.time()
            last_count = current_count
            if (time.time() - start_debounce_time) > 1.0:
                if current_count != debounced_count:
                    debounced_count = current_count
            
            if debounced_count > debounced_last_count:
                start_ptime = time.time()
                total_count = total_count + debounced_count - debounced_last_count
                client.publish("person", json.dumps({"total": total_count}))
            if debounced_count < debounced_last_count:
                pduration = time.time() - start_ptime                   
                client.publish("person/duration", json.dumps({"duration": int(pduration)}))
            client.publish("person", json.dumps({"count": debounced_count}))
            debounced_last_count = debounced_count
            
        # Print output confidence on frame
        print_out_conf(result, out_frame)
        
        # Print inference time on frame
        print_inf_time(inf_t, out_frame)

        if image_flag:
            # Write an output image if `single_image_mode`
            cv2.imwrite('out_image.jpg', out_frame)
        else:
            # Send the frame to the FFMPEG server
            sys.stdout.buffer.write(out_frame)  
            sys.stdout.flush()
            
            
        # Break if escape key pressed
        if key_pressed == 27:
            break
            
    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    # Disconnect from MQTT
    client.disconnect()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
