import cv2
import numpy as np
from scipy.ndimage import zoom
from sklearn.datasets import fetch_olivetti_faces
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import io
import base64
from PIL import Image
faces1=fetch_olivetti_faces()

def string_to_image(base64_string):
    imgdata=base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))


def to_RGB(image):
    return cv2.cvtColor(np.array(image),cv2.COLOR_BGR2RGB)

def create_target(smiles):
    y=np.zeros(faces1.target.shape[0])
    for (start,end) in smiles:
        y[start:end+1]=1
    return y
def detect(string):
    
    img=string_to_image(string)
    img=to_RGB(img)
    print("Fucking here")
    face_cascade=cv2.CascadeClassifier('api/haarcascade_frontalface_default.xml')
    #img=cv2.imread('IMG-20180114-WA0007.jpg')
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)


    #cv2.imshow('face_detected',gray[y:y+h,x:x+w])
    horizontal_offset=0.15*w
    vertical_offset=0.2*h
    #print horizontal_offset,vertical_offset
    extracted_face=gray[int(y+vertical_offset):y+h,int(x+horizontal_offset):int(x-horizontal_offset+w)]
    #cv2.imshow('face_detected',extracted_face)
    new_extracted_face=zoom(extracted_face,(64./extracted_face.shape[0],64./extracted_face.shape[1]))
    new_extracted_face=new_extracted_face.astype('float32')
    new_extracted_face/=float(new_extracted_face.max())
    #cv2.imshow('face_detected',new_extracted_face)
    #print faces1.keys()

        
    smiling=[(10,10),(16,17),(20,21),(23,24),(26,29),(40,49),(51,51),(56,59),(61,63),(65,65),(69,80),(82,82),
             (84,104),(111,114),(116,117),(119,119),(123,123),(128,129),(141,141),(143,144),(146,146),(149,149),
             (153,154),(160,167),(172,174),(177,177),(179,179),(180,185),(186,188),(190,191),(193,197),(200,204),
             (207,207),(209,214),(216,217),(219,219),(224,224),(229,229),(232,234),(240,244),(246,246),(248,249),
             (253,253),(255,265),(267,268),(271,271),(273,278),(297,297),(300,301),
             (303,308),(310,316),(319,329),(333,334),(336,336),(340,350),(361,371),(364,365),
             (367,367),(369,369),(371,372),(376,376),(380,381),(383,399)]

    target_smiles=create_target(smiling)
    X_train,X_test,y_train,y_test=train_test_split(faces1.data,target_smiles,test_size=0.25,random_state=0)
    svc_1=SVC(kernel='linear')
    #cv=KFold(len(y_train),5,shuffle=True,random_state=0)
    #scores=cross_val_score(svc_1,X_train,y_train,cv=cv)
    #print scores
    svc_1.fit(X_train,y_train)
    #print svc_1.score(X_train,y_train)
    #print svc_1.score(X_test,y_test)
    #y_pred=svc_1.predict(X_test)
    #print X_test.shape
    #print y_pred 
    if(svc_1.predict([new_extracted_face.ravel()])[0]==1):
        return 'SMILING'
    else:
        return 'NOT SMILING'
    ##cv2.waitKey(0)
    ##cv2.destroyAllWindows()
    
