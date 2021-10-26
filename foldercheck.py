
import cv2
import  os


img=  cv2.imread('images/apple.jpg')

cv2.imshow('Apple', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

directory= f'results'

os.chdir(directory)

filename= 'new_iamge.jpg'

data= cv2.imwrite(filename, img)

cv2.imshow('c', data)
cv2.waitKey(0)
cv2.destroyAllWindows()