
import os
import h5py
import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import *
import tensorflow as tf
from PIL import Image, ImageTk
from keras import backend as K
from keras.models import load_model
from tkinter.messagebox import showerror, showinfo
from tkinter.filedialog import askopenfilename, asksaveasfilename
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


my_filetypes  = [("Image File",'.jpg'),("Image File",'.png'),("Image File",'.JPEG'),("Image File",'.BMP'), ("Image File",'.PNG')]
model = load_model('models/model_for_gui.h5')



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

        self.master.title("Fetus Recognition Tool")

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
        # Read image
        self.cv_img = cv.imread(os.path.normpath(self.imageName))  # read image as numpy array
        # Get image parameters to check the size of the image
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

            e_img = np.expand_dims(img, axis=0)
            output = model.predict(e_img)
            pred =plot_images(np.argmax(output[0],axis=1).reshape((224,224)))

            plt.figure(figsize = (10,5))
            plt.subplot(1,2,1)  

            plt.title("Original Image", fontsize=14)
            plt.imshow(img)

            plt.subplot(1,2,2)
            plt.title("Fetus Detection", fontsize=14)
            plt.imshow(img)

            plt.imshow(pred, 'jet', interpolation='none', alpha=0.3)

            plt.savefig('Testing.png', format = 'png', dpi = 600)

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
