# import the necessary packages
# if using mac, use pythonw re***.py to run

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import mxnet as mx
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def decodeBox(yolobox, size, dscale):
    i, j, cx, cy, w, h, cls1, cls2, cls3, cls4 = yolobox
    cxt = j * dscale + cx * dscale
    cyt = i * dscale + cy * dscale
    wt = w * size
    ht = h * size
    clsa = np.argmax([cls1, cls2, cls3, cls4])
    if clsa==0:
        cls=1
    elif clsa==1:
        cls=2
    elif clsa==2:
        cls=3
    elif clsa==3:
        cls=20
    return [cxt, cyt, wt, ht, cls]


def bboxdraw(img, label, dscale=32):
    # assert label.shape == (7,7,9)
    thickness = 2
    lineType  = 8
    size = img.shape[1]
    ilist, jlist = np.where(label[:, :, 0] > 0.85)
    #save bbox info
    #bboxl=[]
    # Create figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(np.uint8(img))
    for i, j in zip(ilist, jlist):
        cx, cy, w, h, cls1, cls2, cls3, cls4 = label[i, j, 1:]
        cxt, cyt, wt, ht, cls = decodeBox([i, j, cx, cy, w, h, cls1, cls2, cls3, cls4], size, dscale)
        prob = label[i, j, 0]
        #bboxl.append([cxt, cyt, wt, ht, cls, prob])
        # Create a Rectangle patch
        leftTopPoint = (int(cxt - wt / 2), int(cyt + ht / 2))
        rightButtomPoint = (int(cxt + wt / 2), int(cyt - ht / 2))
        cv2.rectangle(img,leftTopPoint,rightButtomPoint,(0,0,255),thickness,lineType)
        
 
        #rect = patches.Rectangle((cxt - wt / 2, cyt - ht / 2), wt, ht, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        #ax.add_patch(rect)

        name = "unkown"
        if cls == 1:
            name = "car"
        elif cls == 2:
            name = "pedestrian"
        elif cls == 3:
            name = "cyclist"
        elif cls == 20:
            name = "traffic lights"
        s=str(name) + str(prob)
        cv2.putText(img,s,(int(cxt - wt / 2), int(cyt - ht / 2)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,8)
        #plt.text(x=int(cxt - wt / 2), y=int(cyt - ht / 2), s=str(name) + str(prob),
                     #bbox=dict(facecolor='red', alpha=0.5))
    #plt.show()

    return img




# load our serialized model from disk
print("[INFO] loading model...")
#net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# get sym
sym, args_params, aux_params = mx.model.load_checkpoint('drive_full_detect', 318)
logit = sym.get_internals()['logit_output']
mod = mx.mod.Module(symbol=logit, context=mx.cpu())
data_shape=[('data', (1,3,720,1280))]
mod.bind(data_shapes=data_shape)
mod.init_params(allow_missing=False, arg_params=args_params, aux_params=aux_params)

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# for video
#cap = cv2.VideoCapture('video.avi')
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 600.0, (640,480))

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    #frame = vs.read()
    ret, frame = cap.read() 
    # grab the frame dimensions and convert it to a blob
    #(x, y) = (cap.get(3),cap.get(4))
    hi, wi = frame.shape[:2]
    W = int(224 * (wi * 1.0 / hi))
    H = 224
    img_resize = cv2.resize(frame, (W, H))
    #imga = img_resize.transpose((2, 0, 1)).reshape(1, 3, H, W)
    img_test_nd = mx.nd.array(ctx=mx.cpu(),source_array =img_resize.transpose((2,0,1)).reshape(1,3,H,W))
    img_itr = mx.io.NDArrayIter(data =img_test_nd,data_name="data",batch_size=1)
    # predict
    print ("Now is predicting img ")
    out = mod.predict(eval_data=img_itr)
    pred = (out.asnumpy()[0]+1)/2
    # show 
    frame = bboxdraw(img_resize,pred)
    
    # show the output frame
 
    #cv2.imshow("[srcImg]",srcImg)
    
    
    cv2.imshow("Frame", frame)
    #out.write(frame)    
    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    
    # update the FPS counter
    fps.update()
    

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()
#vs.stop()

