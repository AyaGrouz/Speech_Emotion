import requests
import json 

import librosa 

from azureml.core import Workspace, Datastore

# Load the workspace
workspace = Workspace.from_config()

# Get the datastore 

# you can specify the path exactly 
datastore = workspace.datastores['dataset'] 

# Get the file path
file_path = datastore.path('.\anger016.wav')

# Use the file path to load the file
with open(file_path, 'r') as f:
    data = f.read() 
    
# Convert audio data to bytes

def test_audio_service(scoreurl, scorekey , file_path ):
    assert scoreurl != None
    
    if scorekey is None:
        headers = {'Content-Type':'audio/wav'}
    else:
        headers = {'Content-Type':'audio/wav', 'Authorization':('Bearer ' + scorekey)}
    
    with open(file_path, 'rb') as f:
        audio_data = f.read()
        
    resp = requests.post(scoreurl, data=audio_data, headers=headers)
    assert resp.status_code == requests.codes.ok
    assert resp.text != None
    assert resp.headers.get('content-type') == 'application/json'
    assert int(resp.headers.get('Content-Length')) > 0  
