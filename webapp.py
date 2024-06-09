from ultralytics import YOLO
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import math 
# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

# model = YOLO("./runs/detect/train16/weights/last.pt")
model = YOLO("./googlecolab/train/weights/last.pt")

classNames = ["heavy_scratch", "light_scratch", "dent", "watch_face"]


watchFaceSurface = 0
watchDentSurface = 0
watchLScratchSurface = 0
watchHScratchSurface = 0


def gen_frames():
    while True:
        watchFaces = 0
        watchDents = 0
        watchLScratches = 0
        watchHScratches = 0

        global watchFaceSurface
        global watchDentSurface
        global watchLScratchSurface
        global watchHScratchSurface
        global damages
        global formatteddamages
        global watchDamage
        global watchDamagePosX
        global watchDamagePosY
        global watchDamageSize
        global watchFacePosX1
        global watchFacePosX2
        global watchFacePosY1
        global watchFacePosY2
        watchFaceSurface = 0
        watchDentSurface = 0
        watchLScratchSurface = 0
        watchHScratchSurface = 0
        damages = []
        formatteddamages = []
        watchDamage = False
        watchDamagePosX = 0
        watchDamagePosY = 0
        watchDamageSize = 0
        watchFacePosX1 = 0
        watchFacePosX2 = 0
        watchFacePosY1 = 0
        watchFacePosY2 = 0

        dentRatio = 25
        lScratchRatio = 7
        hScratchRatio = 25

        conditionPerc = 0
        success, img = cap.read()
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        results = model(img, stream=True)

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (98, 165, 117), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])
                if classNames[cls] == "watch_face":
                    watchFaces = watchFaces + 1
                    watchFaceSurface = watchFaceSurface + ((x2 - x1) * (y2 - y1) / 2)
                    watchFacePosX1 = x1
                    watchFacePosX2 = x2
                    watchFacePosY1 = y1
                    watchFacePosY2 = y2

                if classNames[cls] == "dent":
                    watchDents = watchDents + 1
                    watchDentSurface = watchDentSurface + ((x2 - x1) * (y2 - y1) * (confidence + 0.5))
                    watchDamagePosX = (x1 + x2) / 2
                    watchDamagePosY = (y1 + y2) / 2
                    watchDamageSize = watchDentSurface
                    damages.append([watchDamagePosX, watchDamagePosY, watchDamageSize])

                if classNames[cls] == "light_scratch":
                    watchLScratches = watchLScratches + 1
                    watchLScratchSurface = watchLScratchSurface + ((x2 - x1) * (y2 - y1) * (confidence + 0.5))
                    watchDamagePosX = (x1 + x2) / 2
                    watchDamagePosY = (y1 + y2) / 2
                    watchDamageSize = watchLScratchSurface
                    damages.append([watchDamagePosX, watchDamagePosY, watchDamageSize])

                if classNames[cls] == "heavy_scratch":
                    watchHScratches = watchHScratches + 1
                    watchHScratchSurface = watchHScratchSurface + ((x2 - x1) * (y2 - y1) * (confidence + 0.5))
                    watchDamagePosX = (x1 + x2) / 2
                    watchDamagePosY = (y1 + y2) / 2
                    watchDamageSize = watchHScratchSurface
                    damages.append([watchDamagePosX, watchDamagePosY, watchDamageSize])


                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_DUPLEX
                fontScale = 1
                color = (252, 252, 252)
                thickness = 2

                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()

        # Calculate condition
        if watchFaces > 0:
            conditionPerc = ((watchFaceSurface - (watchDentSurface * dentRatio) - (watchLScratchSurface * lScratchRatio) - (watchHScratchSurface * hScratchRatio)) / watchFaceSurface) * 100
            for damage in damages:
                if (damage[0] - watchFacePosX1) > 0 and (damage[1] - watchFacePosY1) > 0:
                    posX = (damage[0] - watchFacePosX1) / (watchFacePosX2 - watchFacePosX1)
                    posY = (damage[1] - watchFacePosY1) / (watchFacePosY2 - watchFacePosY1)
                    size = damage[2] / watchFaceSurface
                    formatteddamages.append({'posx': posX, 'posy': posY, 'size': size})

        if conditionPerc < 0:
            conditionPerc = 0

        socketio.emit('watchCondition', {'faces': watchFaces, 'dents': watchDents, 'lscratches': watchLScratches, 'hscratches': watchHScratches, 'condition': conditionPerc, 'damages': formatteddamages})
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')  # concat frame one by one and show result


        # cv2.imshow('Webcam', img)
        # if cv2.waitKey(1) == ord('q'):
        #     break

# cap.release()
# cv2.destroyAllWindows()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    socketio.run(app, allow_unsafe_werkzeug=True)