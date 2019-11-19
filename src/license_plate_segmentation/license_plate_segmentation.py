import numpy as np
import cv2

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

		self.__license_plate_characters_list = []
		"""
			private function
		"""
		self.__preprocess()

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

	def get_license_plate_characters_list(self): 
		"""
			getter license plate characters list
			return license_plate_characters_list
		"""
		return self.__license_plate_characters_list


	"""
		Private Functions
	"""
	def __preprocess(self):
		""" 1. Image to Gray Scale """
		image_HSV = np.zeros((self.__image_height,self.__image_width,3), np.uint8)
		image_HSV = cv2.cvtColor(self.__image_height, cv2.COLOR_BGR2HSV)

		_, _, self.__image_gray = cv2.split(image_HSV)

		""" 2. Get maximize contrast image """
		image_black_hat = np.zeros((self.__image_height,self.__image_width,1), np.uint8)
		image_top_hat = np.zeros((self.__image_height,self.__image_width,1), np.uint8)

		structuring_elements = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		image_top_hat = cv2.morphologyEx(self.__image_gray, cv2.MORPH_TOPHAT, structuring_elements)
		image_black_hat = cv2.morphologyEx(self.__image_gray, cv2.MORPH_BLACKHAT, structuring_elements)

		image_gray_with_top_hat = cv2.add(self.__image_gray, image_top_hat)
		self.__image_maximize_contrast = cv2.subtract(image_gray_with_top_hat, image_black_hat)

		""" 3. Blur Image & Thresh Image """
		image_blur = np.zeros((self.__image_height,self.__image_width,1), np.uint8)
		image_blur = cv2.GaussianBlur(self.__image_maximize_contrast, (5,5), 0)
		self.__image_thresh = cv2.adaptiveThreshold(image_blur, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)

if __name__ == '__main__':
	img = cv2.imread('/Users/zifwang/Desktop/Smart Parking/data/plates/PENNSYLVANIA.jpg')
	licensePlate = license_plate(img)
	character_list = licensePlate.get_license_plate_characters_list()

	print(len(character_list))