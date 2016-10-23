# import the necessary packages
import cv2
from Tkinter import *
root = Tk()
import tkFileDialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
from common import *
from featureNormalizer import FeatureNormalizer
from searchers.acoustic_searcher import AcousticSeacher
from searchers.visual_searcher import VisualSeacher
import os

class UI_class:
	def __init__(self, master, search_path, frame_storing_path):
		self.search_path = search_path
		self.master = master
		self.frame_storing_path = frame_storing_path
		topframe = Frame(self.master)
		topframe.pack()
		
		acoustic_model = load_data(model_folder+str(64)+acoustic_gmm_models)
		normalizer = load_data(train_acoustic_normalizer_path)
		self.acoustic_seacher = AcousticSeacher(acoustic_model, normalizer)

		visual_model = load_data(model_folder+str(18)+visual_gmm_models)
		self.visual_seacher = VisualSeacher(visual_model)

		#Buttons
		Label(topframe).grid(row=0, columnspan=2)
		self.bbutton= Button(topframe, text=" Choose an video ", command=self.browse_query_img)
		self.bbutton.grid(row=1, column=1)
		self.cbutton = Button(topframe, text=" Estimate its venue ", command=self.show_venue_category)
		self.cbutton.grid(row=1, column=2)
		Label(topframe).grid(row=3, columnspan=4)

		self.master.mainloop()


	def browse_query_img(self):

		self.filename = tkFileDialog.askopenfile(title='Choose an Video File').name

		allframes = os.listdir(self.frame_storing_path)
		self.videoname = self.filename.strip().split("/")[-1].replace(".mp4","")

		self.frames = []
		for frame in allframes:
			if self.videoname +"-frame" in frame:
				self.frames.append(self.frame_storing_path + frame)

		COLUMNS = len(self.frames)
		self.columns = COLUMNS
		image_count = 0

		if hasattr(self, 'query_img_frame'):
			self.query_img_frame.pack_forget()
			self.result_img_frame.pack_forget()
		self.query_img_frame = Frame(self.master)
		self.query_img_frame.pack()

		if COLUMNS == 0:
			self.frames.append("none.png")
			print("Please extract the key frames for the selected video first!!!")
			COLUMNS = 1

		for frame in self.frames:

			r, c = divmod(image_count, COLUMNS)
			try:
				im = Image.open(frame)
				resized = im.resize((100, 100), Image.ANTIALIAS)
				tkimage = ImageTk.PhotoImage(resized)

				myvar = Label(self.query_img_frame, image=tkimage)
				myvar.image = tkimage
				myvar.grid(row=r, column=c)

				image_count += 1
				self.lastR = r
				self.lastC = c
			except Exception, e:
				continue

		self.query_img_frame.mainloop()


	def show_venue_category(self):

		self.result_img_frame = Frame(self.master)
		self.result_img_frame.pack()
		if self.columns == 0:
			print("Please extract the key frames for the selected video first!!!")
		else:
			# Please note that, you need to write your own classifier to estimate the venue category to show blow.
			if self.videoname == '1':
			   venue_text = "Home"
			elif self.videoname == '2':
				venue_text = 'Bridge'
			elif self.videoname == '4':
				venue_text = 'Park'

			venue_img = Image.open("venue_background.jpg")
			draw = ImageDraw.Draw(venue_img)

			font = ImageFont.truetype("/Library/Fonts/Arial.ttf",size=66)

			draw.text((50,50), venue_text, (0, 0, 0), font=font)

			resized = venue_img.resize((100, 100), Image.ANTIALIAS)
			tkimage =ImageTk.PhotoImage(resized)

			myvar = Label(self.result_img_frame, image=tkimage)
			myvar.image= tkimage
			myvar.grid(row=self.lastR, column=self.lastC+1)

		self.result_img_frame.mainloop()

window = UI_class(root, search_path='testing/audio/', frame_storing_path='testing/frame/')
