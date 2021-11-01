import cv2
#import facevec
import numpy as np
from keras.preprocessing import image 
from keras.models  import load_model
model = load_model('alert.h5') 
video = cv2.VideoCapture(0)
name = ['Domestic', 'Wild','Human']
    
while(1):
    success, frame = video.read()
    cv2.imwrite("image.jpg",frame)
    img = image.load_img("image.jpg",target_size = (64,64))
    x  = image.img_to_array(img)
    x = np.expand_dims(x,axis = 0)
    pred = np.argmax(model.predict(x),axis=-1)
    p = pred[0]
    print(pred)
    cv2.putText(frame, "predicted  class = "+str(name[p]), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1)
    cv2.imshow("image",frame)
    if cv2.waitKey(1) & 0xFF == ord('a'): 
        break

video.release()
cv2.destroyAllWindows()