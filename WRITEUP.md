# Deploy a People Counter App at the Edge

## Explaining Custom Layers

The Model Optimizer searches the list of known layers for each layer contained in the input model topology before building the Internal Representation, optimizing the model, and producing the Intermediate Representation files.

Custom layers are layers that are not included in the list of known layers. If the topology of the input model contains layers that are not in the list of known layers, the Model Optimizer classifies them as custom.

In order to add custom layers, there are a few differences depending on the original model framework. 
For example, with TensorFlow models, the first option is to register the custom layers as extensions to the Model Optimizer. In this case, the Model Optimizer generates a valid and optimized Intermediate Representation.

The main reason for handling custom layers is that in many cases you need to deploy an Edge AI application starting from a pre-trained and tested model; the ability to manage custom layers gives consistency to the development process.

[Source](https://docs.openvinotoolkit.org/2019_R3.1/_docs_HOWTO_Custom_Layers_Guide.html)

## Comparing Model Performance

In order to compare models before and after conversion to Intermediate Representations I wrote a basic Python script that runs frozen TF models in the same workspace (and with the same hardware) used to test the OpenVino application.

- The sizes of the models pre- and post-conversion are depicted in the table below:

    | Model                 | Size pre- MB     | Size post- MB |
    | :-----------          | :-----------     |:-----------   |
    | SSD Inception V2 COCO | 102,0            | 100,1         |
    | SSD Mobilenet V2 COCO | 69,7             | 67,2          |
    |SSDLite Mobilenet V2 COCO| 19,9           | 17,9          |


- The inference times of the models pre- and post-conversion are depicted in the table below:

    | Model                 | Inf time pre- ms     | Inf time post ms  |
    | :-----------          | :-----------         |:-----------       |
    | SSD Inception V2 COCO | 200-210              | 150-155           |
    | SSD Mobilenet V2 COCO | 140-145              | 70-75             |
    | SSDLite Mobilenet V2 COCO| 95-100            | 30-35             |
    
- I manually compared the accuracy of models pre- and post- conversion. With every model I noticed a drop in accuracy. In general before conversion the detection confidence is more stable, while after conversion is more oscillating and there are false negatives.

## Assess Model Use Cases

A people counter like this can be used for several applications:

- Security: for example if you need to automatically control unauthorized gatherings in public spaces;
- Crowd management: for example if you need to evaluate crowding at public events, such as street concerts; with a people counter you can identify dangerous situation and trigger an alarm;
- Queue management: for example if you need to evaluate the number of people queuing to access a place or an event; knowing the number of people waiting over time can help make predictions and deploy more resources to manage them;
- Health: during the current pandemic, the ability to assess and report gatherings in open and closed spaces is of crucial importance.

## Assess Effects on End User Needs

Images of people and objects strongly depend on lighting: the way in which lighting affects the image of an object involves changes in magnitude, shading, and shadows. The mean level of luminance varies with the overall magnitude of illumination; shading results from the interaction between the orientation of a surface and the light source direction and it may provide information about shape; shadows result from the interaction between a light source and a receiving surface. Then variation in illumination may have consequences for the speed and accuracy of recognition and should be carefully considered when training and using a model. There is another banal fact to take into consideration: it’s hard to see anything in poor light. In this case the use of special lenses and cameras (thermal, infrared and so on) can help.

In camera lenses, longer focal length leads to higher magnification and a narrower angle of view, shorter focal length is associated with lower magnification and a wider angle of view. Consequently focal length determines the image scale, that is the ratio between image and object size (pixel resolution). Accuracy is strongly dependent on the object size-to-pixel ratio, so focal length and image size should be carefully considered. It must also be taken into account that increasing the size of images will increase data communication traffic and computing time.

In general a model with lower intrinsic accuracy could be faster. Since there is a trade-off between accuracy and speed, the most appropriate model for each application depends on the application requirements.

[Source](https://www.sciencedirect.com/science/article/pii/S0042698998000418)

[Source](https://en.wikipedia.org/wiki/Focal_length)

## Model Research

In investigating potential people counter models, I tried each of the following three models.

I started from an SSD model, designed for object detection in real-time.

- Model 1: SSD Inception V2 COCO
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)
  
  - I converted the model to an Intermediate Representation with the following arguments:
  
    ```python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json```
  
  - The model was insufficient for the app because the inference time was too long (between 150 and 155 ms). With a latency so high the application could not be used in real-time with a video stream.
  
  - I tried to change the precision of the IR from FP32 to FP16, but the inference time was pretty much the same (I then figured out that with CPU there is an upscaling to FP32). Eventually I found out that there is a process called INT8 Calibration to lower the precision of a model from FP32 to INT8, but I didn't have enough time to test it, so I decided to switch to another model.

I moved to Mobilenet, designed for mobile and embedded applications.

- Model 2: SSD Mobilenet V2 COCO
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
  
  - I converted the model to an Intermediate Representation with the following arguments:
  
    ```python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json```
  
  - As expected, with this model the inference time was better than the previous one, but still insufficient to achieve a real-time detection with high framerate.
  
  - With this model (and also with the previous one) I noticed that often there was a sort of flickering of the bounding boxes and this resulted in wrong counts. I tried to improve this aspect by writing a simple algorithm that ignores short appearances and disappearances of the bounding boxes.

Eventually I decided to try the Mobilenet model based on SSDlite.

- Model 3: SSDLite Mobilenet V2 COCO

 - [Model Source](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz)
 
 - I converted the model to an Intermediate Representation with the following arguments:
 
     ```python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json```
  
  - This one was the fastest model I tried: in terms of speed for real-time applications it was absolutely acceptable. However, some accuracy issues persisted.
  
  - I tried to adjust the confidence threshold and the algorithm I developed for the previous model in order to obtain a correct total count. I achieved the result, but I wondered if an even better result could be obtained with a model from OpenVino model zoo.
  
## OpenVino model

I chose a model based on Mobilenet, that performed pretty well in my tests.

- Model: person-detection-retail-0013 
 
  - Although the inference time was slightly higher than that obtained with SSDLite Mobilenet V2 COCO (40-45 ms versus 30-35 ms), the accuracy was much better and I didn't noticed all the flickering and detection errors found with previous models.