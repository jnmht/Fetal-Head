
import os
import h5py
import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from tkinter.messagebox import showerror, showinfo
from tkinter.filedialog import askopenfilename, asksaveasfilename
import cv2
import math

import matplotlib

matplotlib.use("TKAgg")
print(matplotlib.get_backend())

from matplotlib import pyplot as plt

blur_filter_size = 5

bnw_threshold = 16
top, right, bottom, left = 10, 350, 225, 590
def segment_image(blurred_frame, threshold=bnw_threshold):

    thresholded_image = cv2.threshold(blurred_frame, threshold, 255, cv2.THRESH_BINARY)[1]

    (version, _, _) = cv2.__version__.split('.')

    (_, cnts, _) = cv2.findContours(thresholded_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    if len(cnts) == 0:
        return
    else:

        segmented_image = max(cnts, key=cv2.contourArea)
        
        return (thresholded_image, segmented_image)

my_filetypes  = [("Image File",'.jpg'),("Image File",'.png'),("Image File",'.JPEG'),("Image File",'.BMP'), ("Image File",'.PNG')]



background = [0,0,0]
fetus = [255, 0, 0]


label_colours = np.array([background, fetus])

def plot_images(predicted_mask):
    r = predicted_mask.copy()
    g = predicted_mask.copy()
    b = predicted_mask.copy()
    for l in range(0,2):
        r[predicted_mask==l]=label_colours[l,0]
        g[predicted_mask==l]=label_colours[l,1]
        b[predicted_mask==l]=label_colours[l,2]

    rgb = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3))
    rgb[:,:,0] = (r/255.0)#[:,:,0]
    rgb[:,:,1] = (g/255.0)#[:,:,1]
    rgb[:,:,2] = (b/255.0)#[:,:,2]
    return rgb

class Window(tk.Frame):


    def __init__(self, master=None):

        tk.Frame.__init__(self, master)   
        self.imageName = ''
        self.image_width = 224 # image size according to the trained data.
        self.image_height = 224 # image size according to the trained data.
                      
        self.master = master


        self.init_window()


    def init_window(self):
    
        self.master.title("Face Recognition Tool")

        self.pack(fill=tk.BOTH, expand=1)


        menubar = Menu(self.master, background='#303030', foreground='#B8B8B8', activebackground='#404040', activeforeground='#B8B8B8', borderwidth=0)
        filemenu = Menu(menubar, tearoff=0, background='#303030', foreground='#B8B8B8', activebackground='#404040', activeforeground='#B8B8B8', borderwidth=0)
        filemenu.add_command(label="Open", command=self.Button_Callback_FileOpen)
     
        filemenu.add_command(label="Exit", command=self.client_exit)
        menubar.add_cascade(label="File", menu=filemenu)

        helpmenu = Menu(menubar, tearoff=0, background='#303030', foreground='#B8B8B8', activebackground='#404040', activeforeground='#B8B8B8', borderwidth=0)
        helpmenu.add_command(label="About", command=self.Button_Callback_About)
        menubar.add_cascade(label="Help", menu=helpmenu)

        self.master.config(menu=menubar)


        openBtn = tk.Button(text='Select Photo', width=15, height=1, command=self.Button_Callback_FileOpen, highlightbackground = '#303030', 
            activeforeground= '#B8B8B8', borderwidth=1, fg='#B8B8B8', bg='#303030', activebackground='#404040')
        testBtn = tk.Button(text='Test', width=15, height=1, command=self.Button_Callback_testBtn, highlightbackground = '#303030', activeforeground= '#B8B8B8',
         borderwidth=1, fg='#B8B8B8', bg='#303030', activebackground='#404040')
       
        openBtn.place(x=300, y=110)
        testBtn.place(x=300, y=155)
        
        w = tk.Label(text="Select photo ..", background='#202020', fg='#B8B8B8')
        w.place(x=20, y=10)
        

        canvas_width = 650
        canvas_height = 450

        self.canvas = tk.Canvas(self,width=canvas_width, height=canvas_height, highlightthickness=0)
        self.canvas.configure(background='#202020')
        self.canvas.pack(side=tk.LEFT)
        self.canvas.create_rectangle(60, 60, self.image_width, self.image_height, fill="white")

    def showImg(self):

        self.cv_img = cv.imread(os.path.normpath(self.imageName))  # read image as numpy array

        imHeight, imWidth, channels = self.cv_img.shape
        imSize = (imWidth, imHeight)

        if(imSize != (self.image_width, self.image_height)):
            self.cv_img = cv.resize(self.cv_img, (self.image_width, self.image_height))


        im = cv.cvtColor(self.cv_img, cv.COLOR_BGR2RGB)
        load = Image.fromarray(im)
        self.render = ImageTk.PhotoImage(load)
        self.canvas.create_image(40,40, anchor=tk.NW, image=self.render)       

    def Button_Callback_FileOpen(self):
        tempImageName = self.imageName
        self.imageName = askopenfilename(title="Please select an Image:", filetypes=my_filetypes)
        if self.imageName == '':
            self.imageName = tempImageName
            pass
        else:
            self.showImg()
            

    def Button_Callback_testBtn(self):
        if self.imageName == '':
            showinfo('Warning', 'Please load image before teting')
        else:
            
            img = self.cv_img
            img = cv.resize(img, (224,224))

            kernel = np.ones((5,5), np.uint8)

            original_frame = img.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            gray = cv2.medianBlur(gray,5)

            blurred_frame = cv2.GaussianBlur(gray, (blur_filter_size, blur_filter_size), 0)
            hand_segment = segment_image(blurred_frame)

            if hand_segment is not None:

                (thresholded_image, segmented_image) = hand_segment

            cv2.drawContours(original_frame, segmented_image, -1, (0, 0, 255))

            plt.figure(figsize = (10,5))
            plt.subplot(1,2,1)  


            plt.title("Original Image", fontsize=14)
            plt.imshow(img)

            #plt.show()

            #plt.subplot(1,4,2)
            #plt.title("Predicted Segmentation", fontsize=14)
            #plt.imshow(pred)
            #print('Shape of Predicted Segmentation Image', pred.shape)

            # Segmented Ground Truth
            #tmp = plot_images(np.argmax(sample_annot, axis=1).reshape(img_h,img_w))
            #plt.subplot(1,4,3)
            #plt.title("Segmentation Ground Truth", fontsize=14)
            #plt.imshow(tmp)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)

            # Overlay
            area = np.unique(original_frame, return_counts = True)[1]
            plt.subplot(1,2,2)
            plt.title("Fetus Detection", fontsize=14)
            plt.xlabel('Area of the fetus: cannot be determined',  labelpad=20)
            plt.imshow(original_frame)
            #plt.imshow(pred, 'jet', interpolation='none', alpha=0.3)

            plt.savefig('Testing_contours.png', format = 'png', dpi = 600)

            plt.show()



    def Button_Callback_About(self):
        # here delet image in folder 
        showinfo('About', 'Version 1.0')

    def client_exit(self):
        self.master.destroy()


def main():
    root = tk.Tk()
    root.geometry("450x300")
    root.resizable(width=False, height=False)
    # Creation of an instance
    app = Window(root)
    # Mainloop 
    root.mainloop()


if __name__== "__main__":
    main()
