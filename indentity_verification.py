import os
requirements_dir = "Requirnments"
# add requirements_dir to the path
os.sys.path.append(requirements_dir)  
import cv2
import re
import numpy as np
import tkinter as tk 
from tkinter import messagebox
from tkinter import PhotoImage
from PIL import Image,ImageTk
import pickle
import pyttsx3
import threading
from time import time

window=tk.Tk()
window.attributes('-fullscreen', True)
window.title("Face Recognition System")

unauthorized_user = False

# paths
meta_data_dir = "meta_data"
images_dir = f"{meta_data_dir}/images"
har_cascade_path = f"{requirements_dir}/haarcascade_frontalface_default.xml"
classifier_path = f"{requirements_dir}/classifier.xml"
details_data_path = f"{meta_data_dir}/details_data.pkl"
bg_image_path = f"{meta_data_dir}/background_image.png"

# Create meta_data directory if it does not exist
if not os.path.exists(meta_data_dir):
    os.makedirs(meta_data_dir)

# Create data directory if it does not exist
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# error beep
def beep():
    window.bell()

class say(threading.Thread):
    def __init__(self, audio ,female=1,rate=180):
        threading.Thread.__init__(self)

        self.audio = audio
        self.voice = int(female)
        self.rate = int(rate)
        
        self.daemon = True
        self.start()

    def run(self):
        try:
            engine = pyttsx3.init('sapi5')
            voices = engine.getProperty('voices')
            engine.setProperty('voice', voices[self.voice].id)
            engine.setProperty('rate', self.rate)
            engine.say(self.audio)
            engine.runAndWait()
            return self.audio
        except:
            pass

# Function to open register window
def open_register_window(): 
        
        global register_window
        register_window = tk.Toplevel()
        register_window.title("Register Window")
        register_window.geometry("500x300+950+50")
        
        l1=tk.Label(register_window,text="Name",font=("Algerian",20))
        l1.place(relx=0.05, rely=0.05)
        t1=tk.Entry(register_window,width=50,bd=5)
        t1.place(relx=0.3, rely=0.05)

        l2=tk.Label(register_window,text="Age",font=("Algerian",20))
        l2.place(relx=0.05, rely=0.2)
        t2=tk.Entry(register_window,width=50,bd=5)
        t2.place(relx=0.3, rely=0.2)

        l3=tk.Label(register_window,text="Gender",font=("Algerian",20))
        l3.place(relx=0.05, rely=0.35)
        t3=tk.Entry(register_window,width=50,bd=5)
        t3.place(relx=0.3, rely=0.35)

        l4=tk.Label(register_window,text="Contact No",font=("Algerian",20))
        l4.place(relx=0.05, rely=0.5)
        t4=tk.Entry(register_window,width=50,bd=5)
        t4.place(relx=0.3, rely=0.5)

        b3=tk.Button(register_window,text="Generate Dataset",font=("Algerian",20),bg='pink',fg='black',command=lambda:generate_dataset(t1,t2,t3,t4))
        b3.place(relx=0.5, rely=0.8, anchor="center")

# Function to train the classifier
def train_classifier(images_dir):
    path = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]
    
    faces = []
    ids = []
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1]) 
        faces.append(imageNp)
        ids.append(id)  
        
    ids = np.array(ids)
   
    # Train and save classifier
    clf = cv2.face.LBPHFaceRecognizer_create()  #Local Binary Patterns Histograms-face recognition algorithm
    clf.train(faces, ids)
    clf.write(classifier_path)
    # messagebox.showinfo('Result', 'Training Dataset Completed')
    say("Data Set Trained", 1, 180)

def generate_dataset(t1,t2,t3,t4):
    global register_window
    id = 0  
    details_data = {} 
    
    if os.path.exists(details_data_path):
        with open(details_data_path, 'rb') as f:
            details_data = pickle.load(f)
        id = max(details_data.keys(), default=0)

    def validate_contact_number(number):
        if len(number) != 10 or not number.isdigit():
            return False
        return True
            
    if t1.get() == "" or t2.get() == "" or t3.get() == "":
        beep()
        say("Please provide complete details", 1, 180)
        # show warning in the register window
        # messagebox.showinfo('Alert', 'Please provide complete details')
    elif not validate_contact_number(t4.get()):
        beep()
        say("Please provide a valid 10 digit contact number", 1, 180)
        # messagebox.showinfo('Alert', 'Please provide a valid 10 digit contact number')
    else:
        say("Please look at the camera to get your face registered", 1, 180)
        face_classifier = cv2.CascadeClassifier(har_cascade_path)
        def face_cropped(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            # scaling factor = 1.3
            # minimum neighbor = 5

            if len(faces) == 0:
                return None
            for (x, y, w, h) in faces:
                cropped_face = img[y:y+h, x:x+w]  # cropping out face from the whole frame
            return cropped_face
        
        cap = cv2.VideoCapture(0)  # 0 for default camera, 1 or -1 is used for external camera
        id += 1
        img_id = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break
            if face_cropped(frame) is not None:
                img_id += 1
                face = cv2.resize(face_cropped(frame), (300, 300))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                
                file_name_path = f"{images_dir}/user.{id}.{img_id}.jpg"
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                
                # Add the captured data to the dictionary
                details_data[id] = {
                    'Name': t1.get(),
                    'Age': int(t2.get()),
                    'Gender': t3.get(),
                    'Contact': t4.get()
                }

                cv2.imshow("Face Cropper", face)

            if cv2.waitKey(1) == 13 or int(img_id) == 100:  # 13 is the ASCII character of Enter
                break

        cap.release()
        cv2.destroyAllWindows()
        # Save the details data dictionary using pickle
        with open(details_data_path, 'wb') as f:
            pickle.dump(details_data, f)
        # messagebox.showinfo('Result', 'Generating Dataset Completed')
        say("data set Generated", 1, 180)
        train_classifier(images_dir) #traning the model

        # after the dataset is generated, close just the register window
        register_window.destroy()

def delete_user_data():
    def delete_selected_user():
        global all_users
        global id_entry

        user_id = int(id_entry.get())
        if user_id in all_users:
            del all_users[user_id]
            os.system(f"rm {images_dir}/user.{user_id}.*.jpg")
            pickle.dump(all_users, open(details_data_path, "wb"))
            say("User data deleted", 1, 180)
            delete_window.destroy()
        else:
            beep()
            say("User not found", 1, 180)
            # messagebox.showinfo('Alert', 'User not found')
    global delete_window
    global all_users
    global id_entry

    # get all the users data and first image of each user
    all_users = pickle.load(open(details_data_path, "rb"))

    # if no user data is present, show a message and return
    if len(all_users) == 0:
        say("No user data found", 1, 180)
        return

    delete_window = tk.Toplevel()
    delete_window.title("Delete User Data")
    delete_window.attributes('-fullscreen', True)
    delete_window.configure(bg='sky blue')

    user_images = [f"{images_dir}/user.{user_id}.1.jpg" for user_id in all_users.keys()]

    for user, image in zip(all_users.values(), user_images):
        user['image'] = image

    # image width = window width / 6
    image_width = int(window.winfo_screenwidth()/6)

    # make a grid of 6x6 to display the users images with their ID written in the center of each image
    for i, user in enumerate(all_users.values()):
        row = i // 6
        col = i % 6
        details = f'{user["Name"]} - {user["Contact"]}'

        current_id = list(all_users.keys())[i]

        img = cv2.imread(user['image'])
        # place the text on the bottom of the image
        cv2.putText(img, details, (10, img.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.resize(img, (image_width, image_width))

        img = Image.fromarray(img)
        img = img.resize((int(window.winfo_screenwidth()/6), (int(window.winfo_screenwidth()/6))))
        img = ImageTk.PhotoImage(img)
        label = tk.Label(delete_window, image=img, text=current_id, font=("Times New Roman", 20), compound=tk.BOTTOM, bg='sky blue')
        # set the label on the image
        label.image = img
        # label.place(x=col*image_width, y=row*image_width)
        label.grid(row=row, column=col, pady=10)

    # below the grid create an input box to take the id of the user to be deleted
    id_label = tk.Label(delete_window, text="Enter the ID of the user to be deleted", font=("Times New Roman", 20), bg='sky blue')
    id_label.grid(row=row+1, column=0, columnspan=6, pady=10)

    id_entry = tk.Entry(delete_window, font=("Times New Roman", 20), bd=5)
    id_entry.grid(row=row+2, column=0, columnspan=6, pady=10)

    delete_button = tk.Button(delete_window, text="Delete", font=("Times New Roman", 20), bg='red', fg='white', command=delete_selected_user)
    delete_button.grid(row=row+3, column=0, columnspan=6, pady=10)

    # create a button to close the delete window
    close_button = tk.Button(delete_window, text="Close", font=("Times New Roman", 20), bg='red', fg='white', command=delete_window.destroy)
    close_button.grid(row=row+4, column=0, columnspan=6, pady=10)


def detect_face():
    # Load details data from pickle file
    with open(details_data_path, 'rb') as f:
        details_data = pickle.load(f)

    say('Please wait, the camera is loading')

    unauthorized_start_time = None

    def draw_boundary(img, classifier, scaleFactor, minNeighbors, clf):
        nonlocal unauthorized_start_time
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

        for (x, y, w, h) in features:
            id, pred = clf.predict(gray_image[y:y+h, x:x+w])
            confidence = int(100 * (1 - pred/300))

            # Display name or unauthorized user above the rectangle
            if id in details_data and confidence > 80:
                name = details_data[id]['Name']
                cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                user_details = details_data[id]
                details_text = f"Name: {user_details.get('Name', 'Unknown')}  Age: {user_details.get('Age', 'Unknown')}  Gender: {user_details.get('Gender', 'Unknown')}  Contact: {user_details.get('Contact', 'Unknown')}"
                cv2.putText(img, details_text, (20, img.shape[0] - 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                # Reset the unauthorized start time
                unauthorized_start_time = None
            else:
                cv2.putText(img, "Unauthorized User", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

                if unauthorized_start_time is None:
                    unauthorized_start_time = time()
                elif time() - unauthorized_start_time > 7:
                    prompt_register_unauthorized_user()
                    unauthorized_start_time = None  # Reset the timer after prompting

    def prompt_register_unauthorized_user():
        # Close the video capture and destroy all windows
        video_capture.release()
        cv2.destroyAllWindows()

        # Ask the user if they want to register the unauthorized user
        response = messagebox.askyesno("Unauthorized User Detected", "Unauthorized user detected. Do you want to register this user?")
        if response:
            open_register_window()
        else:
            # Restart the face detection process
            detect_face()

    # Loading classifier
    faceCascade = cv2.CascadeClassifier(har_cascade_path)
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read(classifier_path)

    cv2.namedWindow("face detection", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("face detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty("face detection", cv2.WND_PROP_TOPMOST, 1)

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, img = video_capture.read()
        draw_boundary(img, faceCascade, 1.1, 10, clf)

        cv2.imshow("face detection", img)
        # set it to close on pressing enter, esc and close button
        if cv2.waitKey(1) == 13 or cv2.waitKey(1) == 27 or cv2.getWindowProperty("face detection", cv2.WND_PROP_VISIBLE) < 1:
            break

    video_capture.release()
    cv2.destroyAllWindows()


def close_program():
    window.destroy()    

if __name__ == "__main__":

    # Set background image
    bg_image = Image.open(bg_image_path)
    bg_image = bg_image.resize((window.winfo_screenwidth(), window.winfo_screenheight()))
    bg_image = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(window, image=bg_image)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    # set foreground and background colors
    fg_color = 'dark blue'
    bg_color = 'sky blue'

    # heading
    heading_label = tk.Label(window, text="Biometric Identity Verification", font=("times new roman", 40, "bold"), fg=fg_color,background=bg_color)
    heading_label.place(relx=0.5, rely=0.95, anchor="center", width=window.winfo_screenwidth())

    # subheading
    subheading_label = tk.Label(window, text="Face Recognition System", font=("Papyrus", 20, "italic","bold"), fg='red',background=bg_color)
    subheading_label.place(relx=0.83, rely=0.87, anchor="center")

    # button properties
    size = 30
    x_pos = 0.45
    y_pos_initial = 0.07
    y_off_set_init = 0.175 
    bg_color = 'light green'
    fg_color = 'navy blue'
    hover_bg_color = 'navy blue'
    hover_fg_color = 'light green'
    font = ("Algerian", size)

    # buttons with labels
    register_button = tk.Button(window, text="Register ", font=font, bg=bg_color, fg=fg_color, command=open_register_window)
    register_button.place(relx=x_pos-0.1, rely=y_pos_initial+y_off_set_init*0, anchor="w")
    register_button.bind("<Enter>", lambda event: hover_label.place(relx=x_pos-0.1, rely=y_pos_initial+y_off_set_init*0+0.05, anchor="w") or hover_label.config(text="Click here to register a new user", fg=hover_fg_color))
    register_button.bind("<Leave>", lambda event: hover_label.place_forget() or hover_label.config(text="", fg=fg_color))

    train_button = tk.Button(window, text="TRAIN", font=font, bg=bg_color, fg=fg_color, command=lambda:train_classifier("data"))
    train_button.place(relx=x_pos+0.02, rely=y_pos_initial+y_off_set_init*1, anchor="w")
    train_button.bind("<Enter>", lambda event: hover_label.place(relx=x_pos+0.02, rely=y_pos_initial+y_off_set_init*1+0.05, anchor="w") or hover_label.config(text="Click here to train the classifier", fg=hover_fg_color))
    train_button.bind("<Leave>", lambda event: hover_label.place_forget() or hover_label.config(text="", fg=fg_color))

    verify_button = tk.Button(window, text="Verify", font=font, bg=bg_color, fg=fg_color, command=detect_face)
    verify_button.place(relx=x_pos+0.05, rely=y_pos_initial+y_off_set_init*2, anchor="w")
    verify_button.bind("<Enter>", lambda event: hover_label.place(relx=x_pos+0.05, rely=y_pos_initial+y_off_set_init*2+0.05, anchor="w") or hover_label.config(text="Click here to verify user identity", fg=hover_fg_color))
    verify_button.bind("<Leave>", lambda event: hover_label.place_forget() or hover_label.config(text="", fg=fg_color))

    delete_button = tk.Button(window, text="DELETE", font=font, bg=bg_color, fg=fg_color, command=delete_user_data)
    delete_button.place(relx=x_pos+0.02, rely=y_pos_initial+y_off_set_init*3, anchor="w")
    delete_button.bind("<Enter>", lambda event: hover_label.place(relx=x_pos+0.02, rely=y_pos_initial+y_off_set_init*3+0.05, anchor="w") or hover_label.config(text="Click here to delete user data", fg=hover_fg_color))
    delete_button.bind("<Leave>", lambda event: hover_label.place_forget() or hover_label.config(text="", fg=fg_color))

    close_button = tk.Button(window, text="Close", font=font, bg=bg_color, fg=fg_color, command=close_program)
    close_button.place(relx=x_pos-0.1, rely=y_pos_initial+y_off_set_init*4, anchor="w")
    close_button.bind("<Enter>", lambda event: hover_label.place(relx=x_pos-0.1, rely=y_pos_initial+y_off_set_init*4+0.05, anchor="w") or hover_label.config(text="Click here to close the program", fg=hover_fg_color))
    close_button.bind("<Leave>", lambda event: hover_label.place_forget() or hover_label.config(text="", fg=fg_color))

    # create an empty label to display the hover text
    hover_label = tk.Label(window, text="", font=("Times New Roman", 20), bg=hover_bg_color, fg=hover_fg_color)

    say("Welcome to the Biometric Identity Verification System", 1, 180)

    window.mainloop()