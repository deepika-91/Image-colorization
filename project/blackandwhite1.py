import cv2
import numpy as np
import argparse
import os
from tkinter import Tk, filedialog, Button, Label, Frame
from tkinter import messagebox

def colorize_image(image_path):
    # Paths to the pre-trained model files
    DIR = r"C:/Users/DEEPIKA/OneDrive/Documents/vspython/AIML"
    prototxt = os.path.join(DIR, r"model/colorization_deploy_v2.prototxt")
    model = os.path.join(DIR, r"model/colorization_release_v2.caffemodel")
    points = os.path.join(DIR, r"model/pts_in_hull.npy")

    # Load model
    print("Loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # Read and process the input image
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Error", "Failed to load image.")
        return

    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    print("Colorizing the image...")
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = (255 * np.clip(colorized, 0, 1)).astype("uint8")

    # Display the images
    cv2.imshow("Original Image", image)
    cv2.imshow("Colorized Image", colorized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def browse_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        colorize_image(file_path)

def main():
    # Set up the GUI
    root = Tk()
    root.title("Image Colorization Tool")
    root.geometry("400x200")
    root.configure(bg="#f0f8ff")

    header_frame = Frame(root, bg="#4682b4")
    header_frame.pack(fill="both")

    label = Label(header_frame, text="Welcome to the Image Colorization Tool!", bg="#4682b4", fg="#ffffff", font=("Helvetica", 16, "bold"), pady=10)
    label.pack()

    sublabel = Label(root, text="Click the button below to select an image.", bg="#f0f8ff", fg="#333333", font=("Helvetica", 12), pady=10)
    sublabel.pack()

    browse_button = Button(root, text="Select Image", command=browse_image, bg="#00bfff", fg="#ffffff", font=("Helvetica", 12, "bold"), padx=20, pady=5)
    browse_button.pack(pady=5)

    exit_button = Button(root, text="Exit", command=root.quit, bg="#dc143c", fg="#ffffff", font=("Helvetica", 12, "bold"), padx=20, pady=5)
    exit_button.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()
