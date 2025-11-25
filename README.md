# E.D.I.T.H. Glasses Proxy Server

Initially, I attempted to do everything on the ESP. However, many issues arose.

## The Issues
### PhotoMath
I reverse engineered the PhotoMath API to solve equations. Initially I attempted to request directly from the ESP, however there were two issues with this approach. 
1. ESP32-S3 does not support TLSv1.3.
2. Response was too large for ESP32's memory

The latter issue was discovered once the proxy server was implemented. The fix to that was stripping the metadata (which was a lot), then sending to the ESP.

### Voice Recognition
The initial approach was training a TensorFlow model on my M4 Pro Macbook Pro, then uploading the tflite file to the ESP. I unfortunately was limited by the memory on the microcontroller. 
The required samples for the machine learning model to accurately predict a voice command would create a file much too large for the ESP to store.

The solution was once again was to implement it on the proxy. This utilizes the Google Speech to Text API, sending back the result.

## Running
To run, you need a Google Cloud Account as well as a Photomath account. 
For Google Cloud, you need a service account with access to the Speech to Text API. Save the JSON to `creds.json` 
For Photomath, you need to figure a way to retrieve your Bearer token. A proxy such as Charles can be of use.

Once all preq's have been fufilled, run `python3 app.py`