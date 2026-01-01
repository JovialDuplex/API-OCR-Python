from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import  Response
from fastapi.middleware.cors import CORSMiddleware
import easyocr
import cv2
import numpy as np
import json
from typing import Optional

app = FastAPI(debug=True)
app.add_middleware(CORSMiddleware, 
               allow_credentials=True, 
               allow_headers=["*"], 
               allow_methods=["*"],
               allow_origins=["*"])

@app.post("/uploadfile")
async def uploadFile(workFile: UploadFile=File(...), 
                    replaceName: Optional[str]=Form(None)):
    
    reader= easyocr.Reader(["fr", "en"])
    
    if workFile.content_type.startswith("image/"):
        
        #load image--------------------------------
        image_bytes = await workFile.read()

        # convert bytes to numpy array -----------------
        np_array = np.frombuffer(image_bytes, np.uint8)

        #decode image-----------------------------------
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        #frame the text on the image------------------
        result = reader.readtext(image)
        data = []
        
        for box, text, conf in result:
            x_coords = [point[0] for point in box]
            y_coords = [point[1] for point in box]

            x = int(min(x_coords))
            y = int(min(y_coords))
            w = int(max(x_coords) - x)
            h = int(max(y_coords) - y)
            
            #Creating mask for remove a text to their position in the image 
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            mask[y: y+h, x:x+w] = 255
        
            image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

            #if you have set the replaceName in the form, then all the text in image will be replace by this text 
            if replaceName:
                image = cv2.putText(image, replaceName, (x, y), cv2.FONT_HERSHEY_SIMPLEX, (w/100), (0, 0, 0), 1)
            
            else:
                bg = cv2.imread("replace1.png")
                bg_resize = cv2.resize(bg, (w, h))
                image[y: y+h, x:x+w] = bg_resize
            
            #save text position of image 
            data.append({
                "text": text,
                "confidence" : float(conf),
                "x": x,
                "y": y,
                "w": w,
                "h": h
            })

        #encode image ------------------------------ 
        _, encode_image = cv2.imencode(".png", image)
        
        #send the image to frontend
        return Response(content=encode_image.tobytes(), media_type="image/png", headers={
            "image-data":  json.dumps(data),
            "Access-Control-Expose-Headers": "image-data",
        })
    
    else :
        print("any image have been send please set an image before sent")
        return {"message": "please send the image file "}


@app.get("/")
def home():
    return {
        "message" : "Welcome To My API OCR Make With Python",
        "infos": {
            "author": "Jovial Duplex",
            "environement": "python3",
            "Compatible-System": "All System are Compatible except the devices that don't have a browser",
            "version": "1.0.0"
        }
    }

