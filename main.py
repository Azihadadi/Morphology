from tkinter import *
from tkinter import filedialog, ttk
from tkinter.font import Font, BOLD
from tkinter.ttk import Style, Notebook

import cv2
import imutils
from PIL import Image, ImageTk
from skimage.filters import threshold_local
import numpy as np

class Root(Tk):
    EROSION = 1
    DILATION = 2
    OPENING = 3
    CLOSING = 4
    BLACK_HAT = 5
    WHITE_HAT = 6
    MORPHOLOGICAL_GRADIENT = 7
    # Styles
    colorHeader = '#2a9d8f'
    colorHeaderTab = '#005580'
    colorContent = "#f0f5f5"
    colorSelected = "#005580"
    colorUnSelected = "#b1cae7"
    colorButton = "#05668d"
    colorExitButton = "red"
    colorButtonText = "white"
    colorActionLabel = "#e76f51"
    colorActionBg = "white"

    actions = {EROSION: "Erosion",
                DILATION:"Dilation",
                OPENING:"Opening",
                CLOSING:"Closing",
                BLACK_HAT: "Black Hat",
                WHITE_HAT: "White Hat",
                MORPHOLOGICAL_GRADIENT: "Morphological Gradient"
                }

    def __init__(self):
        super(Root, self).__init__()
        self.setStylies()
        # headers
        self.var1 = StringVar()
        self.var1.set("Challenge II :Advanced Image Analysis Toolbox")
        self.var2 = StringVar()
        self.var2.set("Mathematical Morphology")
        self.header_up = Label(self, textvariable=self.var1, relief=RAISED, font=self.fontStyleHeader_up,
                               bg=self.colorHeader,
                               fg="white",
                               bd="0",
                               pady="10")
        self.header_down = Label(self, textvariable=self.var2, relief=RAISED, font=self.fontStyleHeader_down,
                                 bg=self.colorHeader,
                                 fg="white",
                                 bd="0",
                                 pady="10")
        # icon
        self.img = ImageTk.PhotoImage(Image.open('images/logo.png'))
        self.imageIcon_label = Label(self, image=self.img)
        self.imageIcon_label.img = self.img

        # Tab Page
        self.tabControl = Notebook(self)
        self.tab_algorithm = Frame(self.tabControl, width=200, height=300)
        self.tab_application = Frame(self.tabControl, width=200, height=300)
        self.tabControl.add(self.tab_algorithm, text="Algorithm")

        # inform frame
        self.info_labelFrame = LabelFrame(self.tab_algorithm, text="What is Mathematical Morphology?", width=450,
                                          height=100,
                                          bd=2,
                                          font=self.fontStyleTabHeader_down, foreground=self.colorHeaderTab,
                                          background="white")
        self.info_labelFrame.grid_propagate(0)

        self.info_label = ttk.Label(self.info_labelFrame,
                                    text="Morphological operations are simple transformations applied to binary or"
                                         "\ngrayscale images.We can use morphological operations to increase the size"
                                         "\nof objects in images as well as decrease them. We can also utilize"
                                         "\nmorphological operations to close gaps between objects as well as open them.",
                                    background="white",
                                    foreground=self.colorSelected, font=self.fontStyleContent_labelFrame, justify=LEFT,
                                    padding=10)

        # Upload Button
        self.width_origin = 0
        self.imageOrigin_label = Label(self.tab_algorithm, image=None)
        self.imageTarget_label = Label(self.tab_algorithm, image=None)
        self.upload_button = Button(self.tab_algorithm, text="Upload", command=self.uploadCallBack, bg=self.colorButton,
                                    fg=self.colorButtonText,
                                    width=10, font=self.fontStyleTabHeader_down, relief=FLAT)

        # Frame of Operations
        self.operation_labelFrame = LabelFrame(self.tab_algorithm, text="Morphological operations", width=570,
                                               height=140,
                                               bd=2,
                                               font=self.fontStyleTabHeader_down, foreground=self.colorHeaderTab,
                                               background="white")
        self.operation_labelFrame.grid_propagate(0)

        # Frame of Images
        self.images_labelFrame = LabelFrame(self.tab_algorithm, text="Original and Result Images", width=800,
                                            height=330,
                                            bd=2,
                                            font=self.fontStyleTabHeader_down, foreground=self.colorHeaderTab,
                                            background="white")
        self.images_labelFrame.grid_propagate(0)
        self.reset = False
        self.var = IntVar()
        self.erosion_radio = ttk.Radiobutton(self.operation_labelFrame, text="Erosion", style='Wild.TRadiobutton',
                                             variable=self.var,
                                             value=self.EROSION,
                                             command=self.resetCounter)
        self.dilation_radio = ttk.Radiobutton(self.operation_labelFrame, text="Dilation", style='Wild.TRadiobutton',
                                              variable=self.var,
                                              value=self.DILATION, command=self.resetCounter)
        self.opening_radio = ttk.Radiobutton(self.operation_labelFrame, text="Opening", style='Wild.TRadiobutton',
                                             variable=self.var,
                                             value=self.OPENING,
                                             command=self.resetCounter)
        self.closing_radio = ttk.Radiobutton(self.operation_labelFrame, text="Closing", style='Wild.TRadiobutton',
                                             variable=self.var,
                                             value=self.CLOSING,
                                             command=self.resetCounter)
        self.black_hat_raidio = ttk.Radiobutton(self.operation_labelFrame, text="Black hat", style='Wild.TRadiobutton',
                                                variable=self.var,
                                                value=self.BLACK_HAT, command=self.resetCounter)
        self.white_hat_raidio = ttk.Radiobutton(self.operation_labelFrame, text="White hat", style='Wild.TRadiobutton',
                                                variable=self.var,
                                                value=self.WHITE_HAT, command=self.resetCounter)
        self.morphological_gradient_radio = ttk.Radiobutton(self.operation_labelFrame, text="Morphological gradient",
                                                            style='Wild.TRadiobutton',
                                                            variable=self.var,
                                                            value=self.MORPHOLOGICAL_GRADIENT,
                                                            command=self.resetCounter)
        self.action_label = Label(self.tab_algorithm, text=None, font=self.fontStyleHeader_down,
                                  bg="white", fg=self.colorActionLabel)
        # Operation Button
        self.counter = 0
        self.operate_botton = Button(self.tab_algorithm, text="Operate", command=self.operateCallBack,
                                     bg=self.colorButton,
                                     fg=self.colorButtonText,
                                     width=10, font=self.fontStyleTabHeader_down, relief=FLAT)
        self.exit_botton = Button(self.tab_algorithm, text="Exit", command=self.exitCallBack,
                                  bg=self.colorExitButton, fg=self.colorButtonText,
                                  width=10, font=self.fontStyleTabHeader_down, relief=FLAT)
        self.footPage_label = Label(self.tab_algorithm,
                                    text="Azadeh Hadadi, Advanced Image Analysis Module, MSCV2, Condorcet University, December 2020",
                                    font=self.fontStyleLabelFooter, fg=self.colorSelected, bg=self.colorContent)
        self.setPositions()

    # functions
    def resetCounter(self):
        self.reset = True
        # reset
        self.action_label['text'] = ''
        self.action_label['bg'] = 'white'

    def uploadCallBack(self):
        self.fileName = filedialog.askopenfilename(initialdir=".\\images", title="Select A File",
                                                   filetype=(("png", "*.png"), ("jpeg", "*.jpg")))
        image_temp = cv2.imread(self.fileName)
        image_temp = cv2.cvtColor(image_temp, cv2.COLOR_BGR2RGB)
        (h, w) = image_temp.shape[:2]
        if h > 200:
            image_temp = imutils.resize(image_temp, height=250)
        self.imageArrayOrigin = Image.fromarray(image_temp)
        self.img = ImageTk.PhotoImage(image=self.imageArrayOrigin)
        self.width_origin = self.img.width()
        self.height_origin = self.img.height()

        self.imageOrigin_label.config(image='')
        self.imageOrigin_label = Label(self.tab_algorithm, image=self.img)
        self.imageOrigin_label.img = self.img
        self.imageOrigin_label.place(x=700, y=100)

        # reset
        self.action_label['text'] = ''
        self.action_label['bg'] = 'white'

        # gray-scale image
        self.gray = cv2.cvtColor(image_temp, cv2.COLOR_BGR2GRAY)
        self.imageArrayTarget = Image.fromarray(self.gray)
        self.imgGrayTarget = ImageTk.PhotoImage(image=self.imageArrayTarget)
        self.imageTarget_label.config(image='')
        self.imageTarget_label = Label(self.tab_algorithm, image=self.imgGrayTarget)
        self.imageTarget_label.img = self.imgGrayTarget
        self.imageTarget_label.place(x=1100, y=100)


    def operateCallBack(self):
        self.counter = self.counter + 1
        if (self.reset == True):
            self.counter = 1
            self.reset = False

        self.imageOrigin = cv2.imread(self.fileName)

        self.kernelSizes = [(3, 3), (5, 5), (7, 7)]

        if (self.var.get() == self.EROSION):
            result = cv2.erode(self.gray.copy(), None, iterations=self.counter)
        elif (self.var.get() == self.DILATION):
            result = cv2.dilate(self.gray.copy(), None, iterations=self.counter)
        elif (self.var.get() == self.OPENING):
            self.kernelSize = self.kernelSizes[(self.counter - 1) %3]
            self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.kernelSize)
            result = cv2.morphologyEx(self.gray, cv2.MORPH_OPEN, self.kernel)
        elif (self.var.get() == self.CLOSING):
            self.kernelSize = self.kernelSizes[(self.counter - 1) %3]
            self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.kernelSize)
            result = cv2.morphologyEx(self.gray, cv2.MORPH_CLOSE, self.kernel)
        elif (self.var.get() == self.BLACK_HAT):
            self.rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
            result = cv2.morphologyEx(self.gray, cv2.MORPH_BLACKHAT, self.rectKernel)
        elif (self.var.get() == self.WHITE_HAT):
            self.rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
            result = cv2.morphologyEx(self.gray, cv2.MORPH_TOPHAT, self.rectKernel)
        elif (self.var.get() == self.MORPHOLOGICAL_GRADIENT):
            self.kernelSize = self.kernelSizes[(self.counter - 1) %3]
            self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.kernelSize)
            result = cv2.morphologyEx(self.gray, cv2.MORPH_GRADIENT, self.kernel)
        # set the action label

        action_str = self.actions.get(self.var.get())
        if(self.var.get() == self.EROSION or self.var.get() == self.DILATION):
            self.action_label['text'] = action_str + ", Iteration: " + str(self.counter)
        elif (self.var.get() == self.OPENING or self.var.get() == self.CLOSING or self.var.get() == self.MORPHOLOGICAL_GRADIENT):
            self.action_label['text'] = action_str + ", Structuring Element Size: " + str(self.kernelSize)
        elif (self.var.get() == self.BLACK_HAT or self.var.get() == self.WHITE_HAT):
            self.action_label['text'] = action_str + ", Structuring Element Size: " + "(13,5)"
        self.action_label['bg'] = self.colorActionBg

        (h, w) = result.shape[:2]
        if h > 200:
            result = imutils.resize(result, height=250)
        self.imageArray = Image.fromarray(result)
        self.targetImage = ImageTk.PhotoImage(image=self.imageArray)
        self.imageTarget_label.config(image='')
        self.imageTarget_label = Label(self.tab_algorithm, image=self.targetImage)
        self.imageTarget_label.img = self.targetImage
        self.imageTarget_label.place(x=1100, y=100)

    def exitCallBack(self):
        self.destroy()

    def setStylies(self):
        # styles
        self.fontStyleHeader_up = Font(family="ARIAL", size=20, weight=BOLD)
        self.fontStyleHeader_down = Font(family="ARIAL", size=13, weight=BOLD)
        self.fontStyleTabHeader_down = Font(family="ARIAL", size=12, weight=BOLD)
        self.fontStyleContent_labelFrame = Font(family="ARIAL", size=12)
        self.fontStyleLabelFooter = Font(family="ARIAL", size=10)

        self.style_tabControl = Style()
        self.style_tabControl.configure('.', background=self.colorContent)
        self.style_tabControl.configure('TNotebook', background="white", tabmargins=[5, 5, 0, 0])
        self.style_tabControl.map("TNotebook.Tab", foreground=[("selected", self.colorSelected)])
        self.style_tabControl.configure('TNotebook.Tab', padding=[10, 4], font=('ARIAL', '13', 'bold'),
                                        foreground=self.colorUnSelected)

        self.style_radioButton = Style()
        self.style_radioButton.configure('Wild.TRadiobutton', background="SystemWindow", foreground=self.colorSelected,
                                         font=self.fontStyleContent_labelFrame)

    def setPositions(self):
        # Position
        self.header_up.place(x=50, y=50)
        self.header_down.place(x=50, y=50)
        self.header_up.pack(fill="x")
        self.header_down.pack(fill="x")
        # icon
        self.imageIcon_label.place(x=0, y=3)
        # tab
        self.tabControl.pack(expand=1, fill="both")
        # operation frame
        self.operation_labelFrame.place(x=10, y=220)
        # images frame
        self.images_labelFrame.place(x=650, y=30)
        # info frame
        self.info_labelFrame.place(x=10, y=30)
        # info frame label
        self.info_label.pack()
        # radio botton
        self.erosion_radio.place(x=0, y=0)
        self.dilation_radio.place(x=180, y=0)
        self.opening_radio.place(x=360, y=0)
        self.closing_radio.place(x=0, y=30)
        self.black_hat_raidio.place(x=180, y=30)
        self.white_hat_raidio.place(x=360, y=30)
        self.morphological_gradient_radio.place(x=0, y=60)
        # action label
        self.action_label.place(x=700, y=70)
        # image origin label
        self.upload_button.place(x=10, y=170)
        # operate botton
        self.operate_botton.place(x=10, y=380)
        # exit button
        self.exit_botton.place(x=150, y=380)
        self.footPage_label.pack(side=BOTTOM, anchor="sw")


root = Root()
root.title("AIA")
root.state('zoomed')
root.mainloop()
