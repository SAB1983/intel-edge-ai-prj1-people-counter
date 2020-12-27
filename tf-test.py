import tensorflow as tf
import time
import cv2

#MODEL_PATH = "models/ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"
#MODEL_PATH = "models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
MODEL_PATH = "models/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb"

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph

def print_out_conf(out):
    count = 0
    tot_confidence = 0
    mean_out_conf = 0
    
    for obj in out[0]:
        if obj[3] > 0:
            tot_confidence += obj[3]
            count += 1
    
    if count > 0:
        mean_out_conf = tot_confidence / count
        
    print("Confidence: {:.1f}%".format(mean_out_conf * 100))

def infer(graph):
    sess = tf.Session(graph=graph)
    inp = graph.get_tensor_by_name('import/image_tensor:0')
    out = graph.get_tensor_by_name('import/detection_boxes:0')
    
    print('Opening video...')
    # Get and open video capture
    cap = cv2.VideoCapture("resources/Pedestrian_Detect_2_1_1.mp4")
    cap.open("resources/Pedestrian_Detect_2_1_1.mp4")
    
    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        # Pre-process the frame
        p_frame = cv2.resize(frame, (224, 224))
        #p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        # Inference start time
        inf_start_t = time.time()
        
        # Perform inference on the frame
        outputs = sess.run(out, feed_dict={inp: p_frame})
        print_out_conf(outputs)
        
        #Compute inference time
        inf_t = time.time() - inf_start_t
        print("Inference time: {:.1f}ms".format(inf_t * 1000))
    return
    
def main():
    print("Loading model...")
    graph = load_graph(MODEL_PATH)
    print("Model loaded.")
    infer(graph)

if __name__ == '__main__':
    main()
