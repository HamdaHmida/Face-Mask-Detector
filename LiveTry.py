import cv2
import numpy as np
from keras.models import load_model 

def Mirror(img):
	rows,cols=img.shape[:2]
	src_pts=np.float32([[0,0], [cols-1,0],[0,rows-1]])
	dst_pts=np.float32([[cols-1,0], [0,0],[cols-1,rows-1]])
	aff_matrix=cv2.getAffineTransform(src_pts,dst_pts)
	img_out=cv2.warpAffine(img,aff_matrix,(cols,rows))
	return img_out

model= load_model('MaskDetector.model')
face_cls= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
source = cv2.VideoCapture(0)
labels_dict = {0: 'No Mask',1: 'Mask' }
color_dict = {1 : (0,0,255), 0:(0,255,0) }

while(True):
    ret,img=source.read()
    img=Mirror(img)
    faces = face_cls.detectMultiScale(img, 1.3, 5)
    
    
    for x,y,w,h in faces :
        face_img=img[y:y+w,x:x+w]
        resized=cv2.resize(face_img, (100, 100))
        reshaped=np.reshape(resized,(1,100,100,3))
        result=model.predict(reshaped)
        
        label = np.argmax(result,axis=1)[0]
        
        cv2.rectangle(img, (x,y), (x+w,y+h), color_dict[label], 2)
        cv2.rectangle(img, (x,y-40), (x+w,y), color_dict[label], -1)
        cv2.putText(img,labels_dict[label],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    cv2.imshow('LIVE',img)
    c=cv2.waitKey(1)
    if c == 27 :
        break
        
        
source.release()
cv2.destroyAllWindows()
