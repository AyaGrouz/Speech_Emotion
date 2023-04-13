import json
import numpy
import joblib
import time
from azureml.core.model import Model  
import soundfile as sf 
import io 
# Provides a way of using operating system dependent functionality. 
import os

# LibROSA provides the audio analysis
import librosa
# Need to implictly import from librosa
import librosa.display

# Import the audio playback widget
import IPython.display as ipd
from IPython.display import Image

# Enable plot in the notebook
import matplotlib.pyplot as plt

# These are generally useful to have around
import numpy as np
import pandas as pd


# To build Neural Network and Create desired Model
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D #, AveragePooling1D
from keras.layers import Flatten, Dropout, Activation # Input, 
from keras.layers import Dense #, Embedding
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder 

# Getting Audio features 
from utils.feature_extraction import get_features_dataframe
from utils.feature_extraction import get_audio_features 

# from inference_schema.schema_decorators \
#     import input_schema, output_schema
# from inference_schema.parameter_types.numpy_parameter_type \
#     import NumpyParameterType


def init():
    global model
    # Load the model from file into a global object
    model_path = Model.get_model_path("speech_emotion_model")
    model = joblib.load(model_path)
    # Print statement for appinsights custom traces:
    print("model initialized" + time.strftime("%H:%M:%S"))


# Inference_schema generates a schema for your web service
# It then creates an OpenAPI (Swagger) specification for the web service
# at http://<scoring_base_url>/swagger.json
# @input_schema('raw_data', NumpyParameterType(input_sample))
# @output_schema(NumpyParameterType(output_sample))
def run(audio_data_bytes, request_headers):    


# Assume `byte_data` is the audio data in bytes
    # Convert the byte data to a file-like object using io.BytesIO
    audio_data, samplerate = sf.read(io.BytesIO(audio_data_bytes))


    try: 
     sampling_rate = 20000  
     demo_mfcc, demo_pitch, demo_mag, demo_chrom = get_audio_features(audio_data,sampling_rate)

     mfcc = pd.Series(demo_mfcc)
     pit = pd.Series(demo_pitch)
     mag = pd.Series(demo_mag)
     C = pd.Series(demo_chrom)
     audio_features = pd.concat([mfcc,pit,mag,C],ignore_index=True)  
     audio_features= np.expand_dims(audio_features, axis=0)
     audio_features= np.expand_dims(audio_features, axis=2) 
    
     result = model.predict(audio_features,batch_size=32,verbose=1) 

     emotions=["anger","disgust","fear","happy","neutral", "sad", "surprise"]
     index = result.argmax(axis=1).item() 
     emotions[index] 

    # Log the input and output data to appinsights:
     info = {
        "input": audio_data,
        "output": emotions[index]
        }
     print(json.dumps(info))
    # Demonstrate how we can log custom data into the Application Insights
    # traces collection.
    # The 'X-Ms-Request-id' value is generated internally and can be used to
    # correlate a log entry with the Application Insights requests collection.
    # The HTTP 'traceparent' header may be set by the caller to implement
    # distributed tracing (per the W3C Trace Context proposed specification)
    # and can be used to correlate the request to external systems.
     print(('{{"RequestId":"{0}", '
           '"TraceParent":"{1}", '
           '"NumberOfPredictions":{2}}}'
           ).format(
               request_headers.get("X-Ms-Request-Id", ""),
               request_headers.get("Traceparent", ""),
               len(result)
    ))

     return {"result": emotions[index]} 
    except Exception as e:
        error = str(e)
        return {"error": error} 
