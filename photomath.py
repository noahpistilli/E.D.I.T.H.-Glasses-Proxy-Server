import requests
import uuid
import json

class PhotoMath:

    # Image dimensions sent by the ESP
    IMAGE_WIDTH = 320
    IMAGE_HEIGHT = 240

    data: dict

    def __init__(self):
        session_id = str(uuid.uuid4()).upper()
        self.data = {
            "view": {"x": 0, "width": 320, "y": 0, "height": 240},
            "configuration": {
                "personalization": {
                    "location": "CA-ON",
                    "preferredMulType": "vertical",
                    "locale": "en",
                    "preferredDivType": "us",
                },
                "metadata": {
                    "scanCounter": 1,
                    "appLanguage": "en",
                    "sessionId": f"scan-{session_id}",
                    "osVersion": "26.0.1",
                    "platform": "IOS",
                    "eventType": "scan",
                    "appVersion": "8.45.0 (1)",
                    "device": "Unknown",
                    "scanId": f"scan-{session_id}",
                },
                "features": {
                    "imageCollectionOptOut": False,
                    "inlineAnimations": "Variant1",
                    "underaged": False,
                    "problemDatabase": True,
                    "debug": False,
                    "allowMissingTranslations": False,
                    "bookpoint": True,
                },
            },
        }

    def request(self, image):
        contents = {
            "image": ("image.jpeg", image, "image/jpeg"),
            "json": (None, json.dumps(self.data), "application/json"),
        }

        headers = {
            "User-Agent": "Photomath/8.45.0.1 (iOS 26.0.1; en; iPhone15,4; Build/23A355)",
            "Accept": "application/json",
            "Authorization": "",
        }


        # Send to photomath and get a result back.
        resp = requests.post(
            "https://rapi.photomath.net/v1/process-image-groups",
            files=contents,
            headers=headers,
        )

        code = resp.status_code
        if code != 200:
            return None
        body = resp.json()

        # Fix it up
        try:
            return json.dumps({
                "success": True,
                "normalizedInput": body["normalizedInput"],
                "preview": {
                    "solution": body["preview"]["solution"],
                },
            })
        except:
            return None

