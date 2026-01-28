from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import cv2, zipfile, tempfile, json, numpy as np, easyocr, io
from simple_lama_inpainting import SimpleLama

app = FastAPI(debug=True)

app.add_middleware(CORSMiddleware, 
                   allow_credentials=True, 
                   allow_methods=["*"], 
                   allow_origins=["*"],
                   allow_headers=["*"],)
@app.post("/")
def hello():
    return {
        "message": "welcome to my API ",
        "Content": "here you can do the inpainting of text with Lama and easyOCR",
        "endpoints": ["/uploadfile"]
    }

@app.post("/uploadfile")
async def uploadFile(file : UploadFile=File(...)):
    
    # loading lama impainting
    lama = SimpleLama()
    
    # loading easyocr
    reader = easyocr.Reader(["fr", "en"])
    if file.content_type.startswith("image/"):
        # create a buffer to read file
        buffer = np.frombuffer(await file.read(), dtype=np.uint8)
        
        #decode image
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        image2 = image.copy()
        
        # read text on the image
        ocr_result = reader.readtext(image)
        
        #creating of mask
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        data = []
        for box, text, conf in ocr_result:
            x_coords = [point[0] for point in box]
            y_coords = [point[1] for point in box]

            x = int(min(x_coords))
            y = int(min(y_coords))
            w = int(max(x_coords) - x)
            h = int(max(y_coords) - y)

            mask = cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)
            
            # this is an OCR image result without inpainting
            image2 = cv2.rectangle(image2, (x, y), (x+w, y+h), (255, 0, 0), 1)

            data.append({
                "text": text,
                "confidence": float(conf),
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            })

        # inpainting
        inpaint_result = lama(image, mask)

        #encoding of data before send
        inpaint_result = np.array(inpaint_result)

        # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        
        success1, inpaint_result_encode = cv2.imencode(".png", inpaint_result)
        success2, ocr_result_encode = cv2.imencode(".png", image2)

        if success1 and success2:
            #zip files before send to client -----------------------

            #creating of temporary zip file
            # tempZip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.writestr("data.json", json.dumps(data))
                zipf.writestr("ocr-result-image.png", ocr_result_encode.tobytes())
                zipf.writestr("inpaint-result-image.png", inpaint_result_encode.tobytes())
            
            zip_buffer.seek(0)

            return StreamingResponse(
                zip_buffer,
                media_type="application/zip",
                headers={
                    "Content-Disposition": "attachment; filename=data.zip"
                }
            )
        # FileResponse(tempZip.name, filename="data.zip", media_type="application/zip")

    else : 
        return {
            "message" : "please select right file before send"
        }
