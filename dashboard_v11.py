import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import math
import pandas as pd
import numpy as np
from hmmlearn import hmm
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from pymavlink import mavutil
import tensorflow as tf
from tensorflow import keras
from ZeroDCE import ZeroDCE
from ultralytics import YOLO
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class App:

    # initializing global variables
    model_path = "googlenet_cnn_only_86acc.h5"
    zero_dce_model_path = "models\\zero_dce.h5"
    model = None
    frame_var=0.0
    pred_var=0.0
    frame_counter=1.0
    com_port="com3"
    baud=57600
    no_lat_long=True
    hmm_show=True
    feed_source=0
    brightness_values = []
    brightness_threshold=80
    #vehicle=mavutil.mavlink_connection(com_port, baud)

    def __init__(self, root):
        self.root = root
        self.root.title("Icily Dashboard")

        # Create frames
        self.frame_upper_left = ttk.Frame(root, width=400, height=300)
        self.frame_upper_right = ttk.Frame(root, width=400, height=300)
        self.frame_lower_left = ttk.Frame(root)
        self.frame_lower_right = ttk.Frame(root)
        self.frame_additional = ttk.Frame(root)

        # Create labels for text and display areas
        self.label_text_webcam = ttk.Label(self.frame_upper_left, text="Webcam Feed")
        self.label_text_thermal = ttk.Label(self.frame_upper_right, text="Processed Feed")
        self.label_webcam = ttk.Label(self.frame_upper_left, text="Feed not available")
        self.label_thermal = ttk.Label(self.frame_upper_right, text="Feed not available")

        # Create buttons
        self.btn_start_webcam = ttk.Button(self.frame_upper_left, text="Start Webcam", command=self.start_webcam)
        self.btn_stop_webcam = ttk.Button(self.frame_upper_left, text="Stop Feed", command=self.stop_feed)

        self.btn_load_feed = ttk.Button(self.frame_upper_left, text="Load Feed", command=self.load_feed)

        # Create editable text area for video path
        self.entry_video_path = ttk.Entry(self.frame_upper_left, width=30)

        # Create editable text area for user input
        self.entry_user_input = ttk.Entry(self.frame_lower_left, width=60)
        self.entry_user_input.bind("<Return>", self.handle_user_input)

        # Create terminal-like text widget
        self.label_output_console = ttk.Label(self.frame_lower_left, text="Output Console")
        self.text_terminal = tk.Text(self.frame_lower_left, wrap="word", height=20, width=50, state=tk.DISABLED)
        self.text_terminal.insert(tk.END, "Welcome to the terminal area!\n")
        self.terminal_scrollbar = ttk.Scrollbar(self.frame_lower_left, orient="vertical", command=self.text_terminal.yview)
        self.text_terminal.config(state=tk.NORMAL)  # Set to NORMAL to allow editing
        self.text_terminal.config(state=tk.DISABLED)  # Set back to DISABLED to make it read-only
        self.text_terminal.config(yscrollcommand=self.terminal_scrollbar.set)
        self.terminal_scrollbar.grid(row=1, column=1, sticky="ns")

        # Create heading for the additional frame
        self.label_additional_heading = ttk.Label(self.frame_additional, text="Model Inference and Stats", font=("TkDefaultFont", 12, "bold"))

        # Create a figure for the line graph with three subplots
        self.figure = plt.Figure(figsize=(5, 6), dpi=100)
        self.plot = self.figure.add_subplot(3, 1, 1)  # Three graphs in a column
        self.plot2 = self.figure.add_subplot(3, 1, 2)
        self.plot3 = self.figure.add_subplot(3, 1, 3)
        self.plot3.set_title('Brightness Over Time')
        self.plot3.set_xlabel('Frame Number')
        self.plot3.set_ylabel('Brightness')
        self.figure.subplots_adjust(hspace=0.7)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame_additional)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, pady=5)

        # Grid layout for upper frames
        self.frame_upper_left.grid(row=0, column=0, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.frame_upper_right.grid(row=0, column=1, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.frame_additional.grid(row=0, column=2, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))  # New frame

        # Grid layout for labels and buttons within upper frames
        self.label_text_webcam.grid(row=0, column=0, columnspan=2)
        self.label_webcam.grid(row=1, column=0, columnspan=2)
        self.btn_start_webcam.grid(row=2, column=0)
        self.btn_stop_webcam.grid(row=2, column=1)
        self.entry_video_path.grid(row=3, column=0, pady=5)
        self.btn_load_feed.grid(row=3, column=1, pady=5)
        self.label_text_thermal.grid(row=4, column=0, columnspan=2)
        self.label_thermal.grid(row=5, column=0, columnspan=2)

        # Additional text widget for the lower right frame
        self.label_additional_console = ttk.Label(self.frame_lower_right, text="Flight_Params Console")
        self.text_additional_terminal = tk.Text(self.frame_lower_right, wrap="word", height=27, width=50, state=tk.DISABLED)
        self.text_additional_terminal.insert(tk.END, "Welcome to the additional terminal area!\n")
        self.text_additional_terminal.config(state=tk.NORMAL)
        self.text_additional_terminal.config(state=tk.DISABLED)

        # Grid layout for additional terminal area
        self.label_additional_console.grid(row=0, column=0, pady=5)
        self.text_additional_terminal.grid(row=1, column=0, padx=10, pady=10)

        # Grid layout for the terminal area
        self.label_output_console.grid(row=0, column=0, pady=5)
        self.text_terminal.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Grid layout for the heading of the additional frame
        self.label_additional_heading.grid(row=0, column=0, pady=5)

        # Grid layout for user input area
        self.entry_user_input.grid(row=2, column=0, pady=5)

        # Place for the lower-left frame
        self.frame_lower_left.place(x=0, y=400)

        # Place for the lower-right frame
        self.frame_lower_right.place(x=420, y=320)

        # Place for the additional frame to the right
        self.frame_additional.place(x=850, y=0)

        # Variable to track if the webcam is running or video is loaded
        self.video_running = False
        self.cap = None

        # Create menu bar
        self.menu_bar = tk.Menu(root)
        self.root.config(menu=self.menu_bar)

        # File menu
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Exit", command=root.quit)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        # Reports menu
        self.reports_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Reports", menu=self.reports_menu)
        self.reports_menu.add_command(label="Show Report", command=self.show_report)
        
        # Export Logs submenu
        self.export_logs_menu = tk.Menu(self.reports_menu, tearoff=0)
        self.reports_menu.add_cascade(label="Export Logs", menu=self.export_logs_menu)
        self.export_logs_menu.add_command(label="fParams", command=self.export_fparams)
        self.export_logs_menu.add_command(label="Terminal Logs", command=self.export_terminal_logs)

        # Authenticate menu
        self.authenticate_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.authenticate_menu.add_command(label="Login", command=self.show_login_window)
        self.authenticate_menu.add_command(label="Register", command=self.register)
        self.menu_bar.add_cascade(label="Authenticate", menu=self.authenticate_menu)

        # Help menu
        self.help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.help_menu.add_command(label="About", command=self.show_about)
        self.menu_bar.add_cascade(label="Help", menu=self.help_menu)

        # Laboratory menu
        self.laboratory_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Laboratory", menu=self.laboratory_menu)

    # Function to detect humans in a frame using YOLOv8
    def detect_humans(self, frame):
        if self.model is None:
            self.model = YOLO('models\\best.pt')
            self.error_to_terminal("No model selected...\n")
            self.print_to_terminal("Using default model...\n")
        results = self.model(frame)[0]
        detected = 0
        coords = []
        for result in results.boxes:
            if result.cls in range(0, 7):
                confidence = result.conf[0]
                if confidence > 0.5:
                    detected += 1
                    coords.append(map(int, result.xyxy[0]))

        return (detected, coords)

    def start_webcam(self):
        if not self.video_running:
            self.video_running = True
            self.btn_start_webcam["state"] = "disabled"
            self.btn_stop_webcam["state"] = "normal"

            # Open the webcam feed
            self.cap = cv2.VideoCapture(self.feed_source)

            while self.video_running:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Resize each frame to 400x300 pixels
                frame = cv2.resize(frame, (400, 300))
                frame_original_size = frame
                dce_frame = cv2.resize(frame_original_size, (600, 400))

                # Display the webcam feed
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.label_webcam.imgtk = imgtk
                self.label_webcam.configure(image=imgtk)

                # Convert to gray frame and increase brightness
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img_thermal = Image.fromarray(frame)
                brightness = gray_frame.mean()
                if(brightness < self.brightness_threshold):
                    self.print_to_terminal(f"Low brightness detected.\nActivating DCE Model for brighter frames.\n")
                    dce_output_frame = self.infer(Image.fromarray(cv2.cvtColor(dce_frame, cv2.COLOR_BGR2RGB)))
                    resized_dce_output_frame = cv2.resize(np.array(dce_output_frame), (400, 300))
                    resized_dce_output_frame = Image.fromarray(resized_dce_output_frame)
                    frame_original_size = resized_dce_output_frame
                
                self.brightness_values.append(brightness)
                self.print_to_terminal(f"Brightness Level: {brightness}\n")

                # Update the third plot with brightness values
                self.plot3.clear()
                self.plot3.plot(self.brightness_values, color='blue')
                self.canvas.draw()
                
                # Using YOLO to predict and display the output frame
                yolo_output = self.detect_humans(frame_original_size)
                self.print_to_terminal(f"Prediction: {str(int(bool(yolo_output[0])))}\n")
                for i in range(yolo_output[0]):
                    x1, y1, x2, y2 = yolo_output[1][i]
                    print(x1, "\t", y1, "\t", x2, "\t", y2)
                    cv2.rectangle(np.array(frame_original_size), (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2image_thermal = cv2.cvtColor(np.array(frame_original_size), cv2.COLOR_BGR2RGB)
                img_thermal = Image.fromarray(cv2image_thermal)
                imgtk_thermal = ImageTk.PhotoImage(image=img_thermal)
                self.label_thermal.imgtk = imgtk_thermal
                self.label_thermal.configure(image=imgtk_thermal)

                # Save frame and pred and updating graph
                self.frame_var = self.frame_counter
                self.frame_counter += 1
                self.pred_var = int(bool(yolo_output[0]))
                self.save_record()

                # Update the Tkinter window
                self.root.update_idletasks()
                self.root.update()

            # self.cap.release()
            # self.cap = None
            self.label_webcam.configure(text="Feed not available")
            self.label_thermal.configure(text="Feed not available")
            self.btn_start_webcam["state"] = "normal"
            self.btn_stop_webcam["state"] = "disabled"
            
    def stop_feed(self):
        self.video_running = False
        self.btn_start_webcam["state"] = "normal"
        self.btn_stop_webcam["state"] = "disabled"

        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.brightness_values = []

        self.label_webcam.configure(text="Feed not available")
        self.label_thermal.configure(text="Feed not available")

    def infer(self, original_image):
        with keras.utils.custom_object_scope({'ZeroDCE': ZeroDCE}):
            self.dce_model = keras.models.load_model(self.zero_dce_model_path)
        image = tf.keras.preprocessing.image.img_to_array(original_image)
        image = image[:, :, :3] if image.shape[-1] > 3 else image
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)
        output_image = self.dce_model(image)
        output_image = tf.cast((output_image[0, :, :, :] * 255), dtype=np.uint8)
        output_image = Image.fromarray(output_image.numpy())
        return output_image

    def load_feed(self):
        if not self.video_running:
            self.frame_counter=1.0
            video_path = self.entry_video_path.get()
            self.cap = cv2.VideoCapture(video_path)
            self.video_running = True
            self.btn_start_webcam["state"] = "disabled"
            self.btn_stop_webcam["state"] = "normal"

            fps = self.cap.get(cv2.CAP_PROP_FPS)
            delay = int(1000 / fps) if fps > 0 else 25  # 25 as default if FPS is not available

            def update():
                ret, frame = self.cap.read()
                if ret:
                    # Resize each frame to 400x300 pixels
                    frame = cv2.resize(frame, (400, 300))
                    frame_original_size = frame
                    dce_frame = cv2.resize(frame, (600, 400))

                    # Display the video feed
                    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(cv2image)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.label_webcam.imgtk = imgtk
                    self.label_webcam.configure(image=imgtk)

                    # Convert to gray model, increase brightness and display
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    img_thermal = Image.fromarray(frame)
                    brightness = gray_frame.mean()
                    if(brightness < self.brightness_threshold):
                        self.print_to_terminal(f"Low brightness detected.\nActivating DCE Model for brighter frames.\n")
                        dce_output_frame = self.infer(Image.fromarray(cv2.cvtColor(dce_frame, cv2.COLOR_BGR2RGB)))
                        resized_dce_output_frame = cv2.resize(np.array(dce_output_frame), (400, 300))
                        resized_dce_output_frame = Image.fromarray(resized_dce_output_frame)
                        frame_original_size = resized_dce_output_frame
                
                    self.brightness_values.append(brightness)
                    self.print_to_terminal(f"Brightness Level: {brightness}\n")

                    # Update the third plot with brightness values
                    self.plot3.clear()
                    self.plot3.plot(self.brightness_values, color='blue')
                    self.canvas.draw()

                    # Using YOLO to predict and display the output frame
                    yolo_output = self.detect_humans(frame_original_size)
                    self.print_to_terminal(f"Prediction: {str(int(bool(yolo_output[0])))}\n")
                    for i in range(yolo_output[0]):
                        x1, y1, x2, y2 = yolo_output[1][i]
                        cv2.rectangle(np.array(frame_original_size), (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2image_thermal = cv2.cvtColor(np.array(frame_original_size), cv2.COLOR_BGR2RGB)
                    img_thermal = Image.fromarray(cv2image_thermal)
                    imgtk_thermal = ImageTk.PhotoImage(image=img_thermal)
                    self.label_thermal.imgtk = imgtk_thermal
                    self.label_thermal.configure(image=imgtk_thermal)

                    # Save frame and pred and updating graph
                    self.frame_var = self.frame_counter
                    self.frame_counter += 1
                    self.pred_var = int(bool(yolo_output[0]))
                    self.save_record()

                    # Update the Tkinter window
                    self.root.update_idletasks()
                    self.root.update()

                    # Schedule the next update after the calculated delay
                    self.root.after(delay, update)
                else:
                    self.stop_feed()

            update()

    def print_to_terminal(self, text):
        self.text_terminal.config(state=tk.NORMAL)
        self.text_terminal.insert(tk.END, text + "\n")
        self.text_terminal.config(state=tk.DISABLED)

    def error_to_terminal(self, text):
        self.text_terminal.config(state=tk.NORMAL)
        self.text_terminal.tag_configure("red", foreground="red")
        self.text_terminal.insert(tk.END, text + "\n", "red")
        self.text_terminal.config(state=tk.DISABLED)

    def print_to_terminal_additional(self, text):
        self.text_additional_terminal.config(state=tk.NORMAL)
        self.text_additional_terminal.insert(tk.END, text + "\n")
        self.text_additional_terminal.config(state=tk.DISABLED)

    def plot_graph(self):
        try:
            # Load data from the CSV file
            self.data = pd.read_csv("files\\data.csv")

            # Plot the data for the first graph
            self.plot.clear()
            self.plot.plot(self.data['frame'], self.data['pred'])
            self.plot.set_xlabel('Frame')
            self.plot.set_ylabel('Prediction')
            self.plot.set_title('Prediction over Frames')

            # Hidden Markov Model (HMM) on pred col.
            hidden_states=[]
            if self.hmm_show:
            # Prepare binary data for HMM
                pred_data = self.data['pred'].values.reshape(-1, 1)

                # Bernoulli HMM for binary data
                model = hmm.CategoricalHMM(n_components=2, n_iter=100, random_state=42)
                model.fit(pred_data)
                hidden_states = model.predict(pred_data)

                # Log state probabilities
                logprob, state_prob = model.score_samples(pred_data)

                self.plot2.clear()
                self.plot2.plot(self.data['frame'], hidden_states, label='HMM State')
                self.plot2.set_xlabel('Frame')
                self.plot2.set_ylabel('HMM State')
                self.plot2.set_title('HMM State over Frames')

                # Also plot the state probabilities
                for i in range(model.n_components):
                    self.plot2.plot(self.data['frame'], state_prob[:, i], label=f'State {i} Probability')

                self.plot2.legend()

            self.canvas.draw()

            # Printing lat and long if command doesn't say otherwise.
            fparams_shown=False
            if not self.no_lat_long:
                while fparams_shown==False:
                    msg = self.vehicle.recv_msg()
                    # self.print_to_terminal_additional(f"---------\n{str(msg)}")
                    # self.print_to_terminal(str(msg))
                    if msg:
                        if msg.get_type() == 'GLOBAL_POSITION_INT':
                            latitude = msg.lat / 1e7 
                            longitude = msg.lon / 1e7
                            altitude = (msg.alt / 1e3)
                            # self.print_to_terminal(f"Latitude: {latitude}, Longitude: {longitude}, Altitude: {altitude} meters\n")
                            # self.print_to_terminal_additional(f"Latitude: {latitude}, Longitude: {longitude}, Altitude: {altitude} meters\n")
                            # fparams_shown=True

                        elif msg.get_type() == 'ATTITUDE':
                            # Assuming compass heading in degrees
                            heading = msg.yaw * 180 / 3.14159 * (1)
                            # self.print_to_terminal(f"Heading: {heading} degrees")
                            self.print_to_terminal_additional(f"Heading: {heading} degrees\n")
                            Earth_Radius=6371000
                            Distance= altitude *(math.tan(30))

                            New_Latitude = latitude + (Distance / Earth_Radius) * (180 / math.pi) * math.cos(heading)
                            New_Longitude = longitude + (Distance / Earth_Radius) * (180 / math.pi) * math.sin(heading)
                            
                            # self.print_to_terminal(f"New Latitude: {New_Latitude}\nNew Longitude: {New_Longitude}\n")
                            self.print_to_terminal_additional(f"New Latitude: {New_Latitude}\nNew Longitude: {New_Longitude}\n")
                            fparams_shown=True

                        else:
                            continue

            # self.print_to_terminal("Graph plotted successfully.")

        except Exception as e:
            self.print_to_terminal(f"Error plotting graph: {e}")

    def handle_user_input(self, event):
        user_input = self.entry_user_input.get().strip()
        if user_input.lower() == "clear":
            self.text_terminal.config(state=tk.NORMAL)
            self.text_terminal.delete(1.0, tk.END)
            self.text_terminal.insert(tk.END, "Terminal cleared.\n")
            self.text_terminal.config(state=tk.DISABLED)
        elif user_input.startswith("set model"):
            self.model_path=user_input[10:]
            self.print_to_terminal(f"-> Model Path: {self.model_path}\n")
        elif user_input.startswith("set com"):
            self.com_port=user_input[8:]
            self.print_to_terminal(f"-> COM Port: {self.com_port}\n")
        elif user_input.startswith("set baud"):
            self.baud=user_input[9:]
            self.print_to_terminal(f"-> Baud Rate: {self.baud}\n")
        elif user_input.startswith("set vehicle"):
            self.vehicle=mavutil.mavlink_connection(self.com_port, self.baud)
            self.print_to_terminal("Vehicle set succesfully!\n")
        elif user_input.startswith("set feed source"):
            feed_source_user_input = user_input[16:]
            if feed_source_user_input=="help":
                self.print_to_terminal("\nFeed Sources available:\n\t0. Local webcam\n\t1. UAV (if set vehicle)\n")
            elif feed_source_user_input.isdigit():
                self.feed_source = int(feed_source_user_input)
                self.print_to_terminal("Feed source changed.")
            else:
                self.error_to_terminal("Invalid command!")
        elif user_input=="toggle display flight_params":
            if self.no_lat_long:
                self.no_lat_long=False
                self.print_to_terminal("Flight Params will be displayed from now on.\n")
            else:
                self.no_lat_long=True
                self.print_to_terminal("Flight Params will NOT be displayed from now on.\n")
        elif user_input.startswith("hmm_stats"):
            if user_input[10:]=="off":
                self.hmm_show=False
                self.print_to_terminal("HMM graph will NOT be displayed.\n")
            else:
                self.hmm_show=True
                self.print_to_terminal("HMM Graph will be displayed.\nPlease note that computing HMM states takes time, and the application might crash if there is a lot of data.\n\n")
        elif user_input == "clear data":
            self.data = pd.DataFrame(columns=['frame', 'pred'])
            self.data.to_csv("files\\data.csv", index=False)
            self.plot.clear()
            self.plot2.clear()
            self.plot3.clear()
            self.canvas.draw()
            self.print_to_terminal("Data cleared successfully.\n")
            self.plot_graph()
        elif user_input.startswith("set dce threshold"):
            self.brightness_threshold = int(user_input[18:])
            self.print_to_terminal(f"Brightness threshold set to {self.brightness_threshold}.\n")
        elif user_input.startswith("activate YOLO"):
            version = user_input[14:]
            if version=='10b':
                self.model = YOLO('models\\yolov10b.pt')
                self.print_to_terminal("YOLOv10b model activated.")
                self.error_to_terminal("Please note that this model is under-development and is currently not recommended.\n")
            elif version=='10m':
                self.model = YOLO('models\\yolov10m.pt')
                self.print_to_terminal("YOLOv10m model activated.")
                self.error_to_terminal("Please note that this model is under-development and is currently not recommended.\n")
            elif version=='10s':
                self.model = YOLO('models\\yolov10s.pt')
                self.print_to_terminal("YOLOv10s model activated.")
                self.error_to_terminal("Please note that this model is under-development and is currently not recommended.\n")
            elif version=='10n':
                self.model = YOLO('models\\yolov10n.pt')
                self.print_to_terminal("YOLOv10n model activated.")
                self.error_to_terminal("Please note that this model is under-development and is currently not recommended.\n")
            elif version=='rec':
                self.model = YOLO('models\\best.pt')
                self.print_to_terminal("Recommended YOLO model activated.\n")
            else:
                self.error_to_terminal("Invalid YOLO model!\n")

        else:
            self.error_to_terminal(f"-> No func. for: {user_input}\n")
        
        # Clear the entry after processing
        self.entry_user_input.delete(0, tk.END)

    def save_record(self):
        try:

            # Appending new record
            new_record = pd.DataFrame({'frame': [self.frame_var], 'pred': [self.pred_var]})
            self.data = pd.concat([self.data, new_record], ignore_index=True)

            self.data.to_csv("files\\data.csv", index=False)

            self.plot_graph()

        except ValueError:
            tk.messagebox.showerror("Error", "Please enter valid numerical values.")

    def show_about(self):
        messagebox.showinfo("About", "This is the alpha version of Icily.\nThis application is currently under heavy development and features will be added/changed at high frequencies.\nA proper documentation will be up on the github repo within some time.\n\nDeveloped by: Icily Team\n")

    def register(self):
        # Placeholder for registration logic
        pass

    def show_report(self):
        # Placeholder for report display logic
        pass

    def export_fparams(self):
        # Get the content of the fParams console
        fparams_content = self.text_additional_terminal.get("1.0", "end-1c")

        # Open a asksaveasfile dialog
        file_path = tk.filedialog.asksaveasfilename(defaultextension=".icll", filetypes=[("Icily Log files", "*.icll")])

        # Write the content to the chosen file
        if file_path:
            with open(file_path, "w") as f:
                f.write(fparams_content)

    def export_terminal_logs(self):
        # Placeholder for exporting terminal logs
        pass

    def logged_in(self):
        self.print_to_terminal("\nLogged in successfully.\n")
        self.login_window.destroy()

    def show_login_window(self):
        self.login_window = tk.Toplevel(self.root)
        self.login_window.title("Login Window")
        
        self.login_window.left_frame = tk.Frame(self.login_window)
        self.login_window.left_frame.pack(side="left", padx=15, pady=10)

        self.login_window.right_frame = tk.Frame(self.login_window)
        self.login_window.right_frame.pack(side="right", padx=25, pady=10)

        # Add company image on the left side
        self.login_window.company_image = tk.PhotoImage(file="logo.png")
        self.login_window.label_company = tk.Label(self.login_window.left_frame, 
                                                image=self.login_window.company_image)
        self.login_window.label_company.pack(pady=10)

        # Add login label and text fields on the right side
        self.login_window.label_login = tk.Label(self.login_window.right_frame, text="Login", font=("Arial", 14, "bold"))
        self.login_window.label_login.pack(pady=10)

        self.login_window.label_userid = tk.Label(self.login_window.right_frame, text="User ID:")
        self.login_window.label_userid.pack()

        self.login_window.entry_userid = tk.Entry(self.login_window.right_frame, justify="center")
        self.login_window.entry_userid.pack(pady=5)

        self.login_window.label_password = tk.Label(self.login_window.right_frame, text="Password:")
        self.login_window.label_password.pack()

        self.login_window.entry_password = tk.Entry(self.login_window.right_frame, show="*", justify="center")
        self.login_window.entry_password.pack(pady=5)

        # Add login button
        self.login_window.button_login = tk.Button(self.login_window.right_frame, text="Login", command=self.logged_in, width=15)
        self.login_window.button_login.pack(pady=10)

# Example of using print_to_terminal
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    app.plot_graph()
    root.mainloop()
