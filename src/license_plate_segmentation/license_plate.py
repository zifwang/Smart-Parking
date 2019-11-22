import numpy as np
import cv2
import math
from possible_characters import possible_characters
from possible_plates import possible_plates
import keras
from keras.models import load_model
# from character_digit_recognition.nn_model import neural_network_model

MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 28
RESIZED_CHAR_IMAGE_HEIGHT = 28

MIN_CONTOUR_AREA = 100

PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5

class license_plate(object):
	def __init__(self, image):
		"""
			init function: initialize license_plate
			argument: image: license plate image
			private variables:
				image:
				image_height:
				image_widht:
				image_area:
				SCALE1:
				SCALE2:
				image_gray
				license_plate_list: list contains characters in the license plate
		"""
		self.__image = image
		self.__image_height = image.shape[0]
		self.__image_width = image.shape[1]
		self.__image_area = self.__image_height * self.__image_width
		self.__image_gray = image

		self.__license_plate_character = ''

	"""
		Getter function
	"""
	def get_input_ori_image(self):
		"""
			getter license plate image
			return image
		"""
		return self.__image

	def get_input_gray_image(self):
		"""
			getter gray license plate image
			return image_gray
		"""
		return self.__image_gray
	
	def get_input_thresh_image(self):
		return self.__image_thresh
	
	def get_input_max_contrast_image(self):
		return self.__image_maximize_contrast

	def get_image_height(self):
		return self.__image_height

	def get_image_width(self):
		return self.__image_width

	def get_image_area(self):
		return self.__image_area

	def get_license_plate_characters(self): 
		"""
			getter license plate characters list
			return license_plate_characters_list
		"""
		return self.__license_plate_character

	def preprocess(self, image_ori):
		height, width, _ = image_ori.shape
		""" 1. Image to Gray Scale """
		image_HSV = np.zeros((height,width,3), np.uint8)
		image_HSV = cv2.cvtColor(image_ori, cv2.COLOR_BGR2HSV)

		_, _, image_gray = cv2.split(image_HSV)

		""" 2. Get maximize contrast image """
		image_black_hat = np.zeros((height,width,1), np.uint8)
		image_top_hat = np.zeros((height,width,1), np.uint8)

		structuring_elements = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		image_top_hat = cv2.morphologyEx(image_gray, cv2.MORPH_TOPHAT, structuring_elements)
		image_black_hat = cv2.morphologyEx(image_gray, cv2.MORPH_BLACKHAT, structuring_elements)

		image_gray_with_top_hat = cv2.add(image_gray, image_top_hat)
		self.__image_maximize_contrast = cv2.subtract(image_gray_with_top_hat, image_black_hat)

		""" 3. Blur Image & Thresh Image """
		image_blur = np.zeros((height,width,1), np.uint8)
		image_blur = cv2.GaussianBlur(self.__image_maximize_contrast, (5,5), 0)
		image_thresh = cv2.adaptiveThreshold(image_blur, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)

		return image_gray, image_thresh
	
	def detect_possible_plates(self, image_ori):
		possbile_plate_list = []

		height, width, _ = image_ori.shape

		grayImage = np.zeros((height,width,1), np.uint8)
		threshImage = np.zeros((height,width,1), np.uint8)

		grayImage, threshImage = self.preprocess(image_ori)
		self.__image_gray = grayImage
		self.__image_thresh = threshImage

		possible_characters_in_scene = self.__find_possible_chars_in_scene()

		listOfListsOfMatchingCharsInScene = self.__findListOfListsOfMatchingChars(possible_characters_in_scene)

		for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
			possible_plate = self.__extractPlate(self.__image, listOfMatchingChars)
			if possible_plate.image_license_plate is not None:
				possbile_plate_list.append(possible_plate)

		return possbile_plate_list


	def __find_possible_chars_in_scene(self):
		possible_char_list = []

		possible_char_counter = 0

		image_thresh_copy = self.__image_thresh.copy()

		# find all contours
		_, contours, _ = cv2.findContours(image_thresh_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   

		for contour in contours:
			possible_chars = possible_characters(contour)
			if self.__is_possible_character(possible_chars):
				possible_char_counter += 1
				possible_char_list.append(possible_chars)
		
		return possible_char_list

	def __is_possible_character(self, possible_characters):
		if (possible_characters.intBoundingRectArea > MIN_PIXEL_AREA and
        	possible_characters.intBoundingRectWidth > MIN_PIXEL_WIDTH and possible_characters.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        	MIN_ASPECT_RATIO < possible_characters.fltAspectRatio and possible_characters.fltAspectRatio < MAX_ASPECT_RATIO):
			return True

		return False
	
	def __distance_between_characters(self, a, b):
		dist_x = abs(a.intCenterX - b.intCenterX)
		dist_y = abs(a.intCenterY - b.intCenterY)

		return math.sqrt(dist_x**2+dist_y**2)

	def __angle_between_characters(self, a, b):
		fltAdj = float(abs(a.intCenterX - b.intCenterX))
		fltOpp = float(abs(a.intCenterY - b.intCenterY))

		if fltAdj != 0.0:                           
			fltAngleInRad = math.atan(fltOpp / fltAdj)     
		else:
			fltAngleInRad = 1.5708                            

		return fltAngleInRad * (180.0 / math.pi)      

	def __findListOfListsOfMatchingChars(self, listOfPossibleChars):
		listOfListsOfMatchingChars = []      

		for possibleChar in listOfPossibleChars:                        
			listOfMatchingChars = self.__findListOfMatchingChars(possibleChar, listOfPossibleChars)        

			listOfMatchingChars.append(possibleChar)                

			if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:     
				continue                            

			listOfListsOfMatchingChars.append(listOfMatchingChars)     

			listOfPossibleCharsWithCurrentMatchesRemoved = []

			listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

			recursiveListOfListsOfMatchingChars = self.__findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)      

			for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:      
				listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)            

			break     

		return listOfListsOfMatchingChars

	def __findListOfMatchingChars(self, possibleChar, listOfChars):
		listOfMatchingChars = []                

		for possibleMatchingChar in listOfChars:               
			if possibleMatchingChar == possibleChar:    
				continue                                

			fltDistanceBetweenChars = self.__distance_between_characters(possibleChar, possibleMatchingChar)

			fltAngleBetweenChars = self.__angle_between_characters(possibleChar, possibleMatchingChar)

			fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)

			fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
			fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)

			if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
				fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
				fltChangeInArea < MAX_CHANGE_IN_AREA and
				fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
				fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):

				listOfMatchingChars.append(possibleMatchingChar)       

		return listOfMatchingChars                  

	def __extractPlate(self, imgOriginal, listOfMatchingChars):
		plate = possible_plates()
		listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # sort chars from left to right based on x position

		# calculate the center point of the plate
		fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
		fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0

		ptPlateCenter = fltPlateCenterX, fltPlateCenterY

		# calculate plate width and height
		intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

		intTotalOfCharHeights = 0

		for matchingChar in listOfMatchingChars:
			intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight

		fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

		intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

		# calculate correction angle of plate region
		fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
		fltHypotenuse = self.__distance_between_characters(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
		fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
		fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

		# pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
		plate.rr_loc_plate_in_scene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

		# get the rotation matrix for our calculated correction angle
		rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

		height, width, _ = imgOriginal.shape      # unpack original image width and height

		imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))       # rotate the entire image

		imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

		plate.image_license_plate = imgCropped         # copy the cropped plate image into the applicable member variable of the possible plate

		return plate

	def __findPossibleCharsInPlate(self,imgGrayscale, imgThresh):
		listOfPossibleChars = []                        # this will be the return value
		contours = []
		imgThreshCopy = imgThresh.copy()

				# find all contours in plate
		_, contours, _ = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

		for contour in contours:                      
			possibleChar = possible_characters(contour)
			if self.__is_possible_character(possibleChar):              
				listOfPossibleChars.append(possibleChar)      

		return listOfPossibleChars

	def __removeInnerOverlappingChars(self,listOfMatchingChars):
		listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)                # this will be the return value

		for currentChar in listOfMatchingChars:
			for otherChar in listOfMatchingChars:
				if currentChar != otherChar:        # if current char and other char are not the same char . . .
																				# if current char and other char have center points at almost the same location . . .
					if self.__distance_between_characters(currentChar, otherChar) < (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
						if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:         # if current char is smaller than other char
							if currentChar in listOfMatchingCharsWithInnerCharRemoved:              # if current char was not already removed on a previous pass . . .
								listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)         # then remove current char
						else:                                                                       # else if other char is smaller than current char
							if otherChar in listOfMatchingCharsWithInnerCharRemoved:                # if other char was not already removed on a previous pass . . .
								listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)           # then remove other char

		return listOfMatchingCharsWithInnerCharRemoved

	def detect_characters_in_license_plate(self, possible_plate_list):
		plate_counter = 0
		image_contours = None
		contours = []

		if len(possible_plate_list) == 0:
			return possible_plate_list
		
		for possible_plate in possible_plate_list:
			possible_plate.image_gray, possible_plate.image_thresh = self.preprocess(possible_plate.image_license_plate)
			possible_plate.image_thresh = cv2.resize(possible_plate.image_thresh, (0, 0), fx = 1.6, fy = 1.6)
			_, possible_plate.image_thresh = cv2.threshold(possible_plate.image_thresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
			listOfPossibleCharsInPlate = self.__findPossibleCharsInPlate(possible_plate.image_gray, possible_plate.image_thresh)
			listOfListsOfMatchingCharsInPlate = self.__findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)

			if len(listOfListsOfMatchingCharsInPlate) == 0:
				possible_plate.possible_chars = ''
				continue

			for i in range(0, len(listOfListsOfMatchingCharsInPlate)):                            
				listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX)       
				listOfListsOfMatchingCharsInPlate[i] = self.__removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])   
			
			intLenOfLongestListOfChars = 0
			intIndexOfLongestListOfChars = 0

			for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
				if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
					intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
					intIndexOfLongestListOfChars = i

			longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]
			possible_plate.possible_chars = self.__recognizeCharsInPlate(possible_plate.image_thresh,longestListOfMatchingCharsInPlate)


		return possible_plate_list
			
	def __recognizeCharsInPlate(self, imgThresh, listOfMatchingChars):
		license_plate_characters = ''           
		height, width = imgThresh.shape

		imgThreshColor = np.zeros((height, width, 3), np.uint8)

		listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # sort chars from left to right

		cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)                     # make color version of threshold image so we can draw contours in color on it

		keras_model = load_model('../model/cnn_classifier.h5')
		alphabets_dict = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A', 11:'B', 12:'C', 13:'D',
                      14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K', 21:'L', 22:'M', 23:'N', 24:'O', 25:'P', 26: 'Q', 
                      27:'R', 28:'S', 29:'T', 30:'U', 31:'V', 32:'W', 33:'X', 34:'Y', 35: 'Z'}
    
		for currentChar in listOfMatchingChars:                                         # for each char in plate
			pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
			pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

			cv2.rectangle(imgThreshColor, pt1, pt2, (0.0, 255.0, 0.0), 2)           # draw green box around the char

			imgROI = imgThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
							currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

			imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))           # resize image, this is necessary for char recognition
			
			npaROIResized = imgROIResized.reshape((1,RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT,1))     
			
			prediction = keras_model.predict(npaROIResized)
			predict_char = alphabets_dict[np.argmax(prediction.T)]
			license_plate_characters = license_plate_characters + predict_char
			# print("The digit in the following figure is: ", alphabets_dict[np.argmax(prediction.T)])
			
			# cv2.imshow('char',imgROIResized)
			# cv2.waitKey(0)

		return license_plate_characters

	def license_plate_detector(self, image):
		possbile_plate_list = self.detect_possible_plates(image)
		possbile_plate_list = self.detect_characters_in_license_plate(possbile_plate_list)

		if len(possbile_plate_list) == 0:
			print("No license plates were detected\n")
		else:
			possbile_plate_list.sort(key = lambda possible_plates: len(possible_plates.possible_chars), reverse = True)
			licPlate = possbile_plate_list[0]
			self.__license_plate_character = licPlate.possible_chars
			cv2.imshow("imgPlate", licPlate.image_license_plate)
			cv2.waitKey(0)
			cv2.imshow("imgThresh", licPlate.image_thresh)
			cv2.waitKey(0)
			cv2.imshow("imagegray", licPlate.image_gray)
			cv2.waitKey(0)





if __name__ == '__main__':
	img = cv2.imread('/Users/zifwang/Desktop/Smart Parking/data/plates/UTAH.jpg')
	licensePlate = license_plate(img)
	licensePlate.license_plate_detector(img)
	print(licensePlate.get_license_plate_characters())

