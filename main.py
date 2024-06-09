from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8s.yaml")  # build a new model from scratch



# Train the model
# model = YOLO("yolov8s.pt") # load a pretrained model (recommended for training)
# model.train(data="config.yaml", epochs=1200, imgsz=2560, rect=True, batch=1, workers=0, pretrained=True, save=True, save_period=50, patience=0)  # train the model

# Resume training of the model
model = YOLO("./runs/detect/train19/weights/epoch100.pt") # load a self-pretrained model
model.train(data="config.yaml", epochs=1200, imgsz=2560, rect=True, batch=1, workers=0, pretrained=True, save=True, save_period=50, patience=0, lr0=0.001, augment=True, resume=True)  # train the model



# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format