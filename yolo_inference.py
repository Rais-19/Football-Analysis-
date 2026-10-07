from ultralytics import YOLO

model=YOLO("model/best.pt") #load a pretrained model (recommended for training)
results=model.predict(r"D:\Computer Vision\Football Analysis\input_videos\video.mp4",save=True)
print(results[0]) #first frame
print("====================================")
for box in results[0].boxes:
    print(box)
