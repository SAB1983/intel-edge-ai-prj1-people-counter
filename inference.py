#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.plugin = None
        self.network = None
        self.exec_network = None
        self.input_blob = None
        self.output_blob = None
        self.infer_request_handle = None

    def load_model(self, model, device="CPU", cpu_extension=None):
        '''
        Load the model IR files.
        '''
        # Load the model
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        # Initialize the plugin
        log.info("Initialize the plugin...")
        self.plugin = IECore()
        
        # Add a CPU extension, if applicable
        if "CPU" in device and cpu_extension:
            log.info("Add CPU extension...")
            self.plugin.add_extension(cpu_extension, device)
        
        # Read the IR
        log.info("Read the IR...")
        self.network = IENetwork(model=model_xml, weights=model_bin)
        
        # Check whether all layers are supported
        supported_layers = self.plugin.query_network(self.network, device)
        not_supported_layers = [layer for layer in self.network.layers.keys() \
                                if layer not in supported_layers]
        if(len(not_supported_layers) > 0):
            log.error("Following layers are not supported:\n {}".
                      format(', '.join(not_supported_layers)))
            sys.exit(1)

        # Load the IR into the plugin
        log.info("Load IR into the plugin...")
        self.exec_network = self.plugin.load_network(self.network, device)
        
        # Get the input layer
        log.info("Get the input layer...")
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        
        return

    def get_input_shape(self):
        '''
        Gets the input shape of the network.
        '''
        log.info("Gets the input shape of the network...")
        return self.network.inputs[self.input_blob].shape

    def exec_net(self, request_id, frame):
        '''
        Makes an asynchronous inference request.
        '''
        self.infer_request_handle = self.exec_network.start_async(request_id=request_id, \
                                      inputs={self.input_blob: frame})
        
        return

    def wait(self, request_id):
        '''
        Checks the status of the inference request.
        '''
        status = self.exec_network.requests[request_id].wait(-1)
        return status

    def get_output(self, request_id):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        return self.exec_network.requests[request_id].outputs[self.output_blob]
