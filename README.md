Real-Time Object Detection with Python GUI

A real-time object detection system using a webcam feed, powered by YOLOv5 and built with OpenCV and a user-friendly Tkinter GUI. This project detects common objects in live video, displays bounding boxes, object counts, and FPS — all within an interactive desktop app.


---

🚀 Features

🎥 Real-time webcam object detection

📦 Powered by pre-trained YOLOv5s (from Ultralytics)

🖥️ Tkinter GUI for live video display and interaction

📋 Object label list with count (e.g., person x 2)

⚡ Live FPS counter overlay

📸 Save screenshot option

🧵 Threaded camera capture for smooth GUI



---

📦 Requirements

Install all dependencies with:

pip install -r requirements.txt

Main Libraries:

torch
opencv-python
pillow
pandas


Also clone YOLOv5:

git clone https://github.com/ultralytics/yolov5


---

🛠️ How to Run

python your_script_name.py

> Make sure yolov5 folder is in the same directory.




---

📁 Project Structure

├── yolov5/ 

├── detect_webcam.py

├── requirements.txt

└── README.md


---

📷 Screenshot

![Screenshot (27)](https://github.com/user-attachments/assets/12ebd7e2-2079-4e0f-9b37-29ecc55f25a2)


---


📌 Use Cases

AI learning and demos

Smart classroom apps

Simple surveillance system

Entry-level AI projects



---

📚 References

Ultralytics YOLOv5

PyTorch

OpenCV

Tkinter Docs
