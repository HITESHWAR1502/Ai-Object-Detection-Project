import torch
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import datetime
import time
from collections import Counter
model = torch.hub.load('yolov5', 'yolov5s','yolov5v', source='local')
class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title(" Object Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg="#121212")
        self.running = False
        self.cap = None
        self.fps = 0
        self.last_time = time.time()
        self.create_widgets()

    def create_widgets(self):
        header = tk.Label(self.root, text="Webcam Object Detection", font=("Segoe UI", 24, "bold"), bg="#121212", fg="white")
        header.pack(pady=10)

        self.content_frame = tk.Frame(self.root, bg="#121212")
        self.content_frame.pack()

        
        self.video_label = tk.Label(self.content_frame, bg="#1e1e1e")
        self.video_label.grid(row=0, column=0, padx=10, pady=10)

        
        list_frame = tk.LabelFrame(self.content_frame, text="Detected Objects", font=("Segoe UI", 12), fg="white", bg="#1e1e1e", bd=2)
        list_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        self.obj_listbox = tk.Listbox(list_frame, font=("Consolas", 12), width=30, height=20, bg="#252526", fg="white", selectbackground="#007acc")
        self.obj_listbox.pack()

        
        ctrl_frame = tk.Frame(self.root, bg="#121212")
        ctrl_frame.pack(pady=10)

        self.start_btn = tk.Button(ctrl_frame, text="▶ Start Detection", command=self.start_detection, font=("Segoe UI", 12), bg="#0b8457", fg="white", width=20)
        self.start_btn.grid(row=0, column=0, padx=10)

        self.stop_btn = tk.Button(ctrl_frame, text="⏹ Stop Detection", command=self.stop_detection, font=("Segoe UI", 12), bg="#b00020", fg="white", width=20)
        self.stop_btn.grid(row=0, column=1, padx=10)

        self.save_btn = tk.Button(ctrl_frame, text="📸 Save Screenshot", command=self.save_screenshot, font=("Segoe UI", 12), bg="#007acc", fg="white", width=20)
        self.save_btn.grid(row=0, column=2, padx=10)

    def start_detection(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        self.running = True
        threading.Thread(target=self.detect_objects, daemon=True).start()

    def stop_detection(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image='')
        self.obj_listbox.delete(0, tk.END)

    def save_screenshot(self):
        if hasattr(self, 'current_frame'):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            messagebox.showinfo("Saved", f"Screenshot saved as {filename}")

    def detect_objects(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(img_rgb)
            results.render()

            # FPS Calculation
            curr_time = time.time()
            self.fps = 1 / (curr_time - self.last_time)
            self.last_time = curr_time

            result_frame = results.ims[0]
            result_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
            cv2.putText(result_frame, f"FPS: {self.fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.current_frame = result_frame

            img = Image.fromarray(result_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            self.obj_listbox.delete(0, tk.END)
            labels = results.pandas().xyxy[0]['name'].tolist()
            label_counts = Counter(labels)
            for label, count in sorted(label_counts.items()):
                self.obj_listbox.insert(tk.END, f"• {label} x {count}")

        if self.cap:
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOApp(root)
    root.mainloop()
