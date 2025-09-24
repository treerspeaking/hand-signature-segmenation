from roboflow import Roboflow
rf = Roboflow(api_key="X20MwEffAbCO1lGRNtu6")
project = rf.workspace("le-viet-tung").project("sure-ey9zm")
version = project.version(1)
dataset = version.download("yolov11")
                