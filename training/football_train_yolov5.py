from roboflow import Roboflow
import shutil
import subprocess

rf = Roboflow(api_key="si37mfEYdmu1WjoDm3VS")
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(1)
dataset = version.download("yolov5")

shutil.move('football-players-detection-1/train',
            'football-players-detection-1/football-players-detection-1/train')

shutil.move('football-players-detection-1/test',
            'football-players-detection-1/football-players-detection-1/test')

shutil.move('football-players-detection-1/valid',
            'football-players-detection-1/football-players-detection-1/valid')

subprocess.run([
    "yolo",
    "task=detect",
    "mode=train",
    "model=yolov5x.pt",
    f"data={dataset.location}/data.yaml",
    "epochs=100",
    "imgsz=640"
])