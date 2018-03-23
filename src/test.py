import mxnet as mx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from Symbol.symbol import get_resnet_model
from Symbol.symbol import  YOLO_loss
from data_ulti import get_iterator


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
    size = img.shape[1]
    ilist, jlist = np.where(label[:, :, 0] > 0.85)
    #save bbox info
    bboxl=[]
    # Create figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(np.uint8(img))
    for i, j in zip(ilist, jlist):
        cx, cy, w, h, cls1, cls2, cls3, cls4 = label[i, j, 1:]
        cxt, cyt, wt, ht, cls = decodeBox([i, j, cx, cy, w, h, cls1, cls2, cls3, cls4], size, dscale)
        prob = label[i, j, 0]
        bboxl.append([cxt, cyt, wt, ht, cls, prob])
        # Create a Rectangle patch
        rect = patches.Rectangle((cxt - wt / 2, cyt - ht / 2), wt, ht, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        name = "unkown"
        if cls == 1:
            name = "car"
        elif cls == 2:
            name = "pedestrian"
        elif cls == 3:
            name = "cyclist"
        elif cls == 20:
            name = "traffic lights"
        plt.text(x=int(cxt - wt / 2), y=int(cyt - ht / 2), s=str(name) + str(prob),
                     bbox=dict(facecolor='red', alpha=0.5))
    #plt.show()

    return np.float64(bboxl).tolist()
if __name__ == "__main__":
    # prepare test data
    imgroot = "./DATA/testing/testing/"
    imglist1 = []

    for i in range(2000):
        imglist1.append("{}".format(70091 + i) + ".jpg")

    imgname = imglist1[0]
    img = cv2.imread(imgroot + imgname)
    hi, wi = img.shape[:2]
    W = int(224 * (wi * 1.0 / hi))
    H = 224
    img_resize = cv2.resize(img, (W, H))
    imga = img_resize.transpose((2, 0, 1)).reshape(1, 3, H, W)
    imgb = imga
    print ("Now is predicting img {}".format(imgname))

    for i in range(1,(len(imglist1))):

        imgname = imglist1[i]
        img = cv2.imread(imgroot + imgname)
        hi, wi = img.shape[:2]
        W = int(224 * (wi * 1.0 / hi))
        H = 224
        img_resize1 = cv2.resize(img, (W, H))
        imga = img_resize1.transpose((2, 0, 1)).reshape(1, 3, H, W)
        print ("Now is predicting img {}".format(imgname))
        imgb = np.vstack([imgb, imga])


    img_nd = mx.nd.array(ctx=mx.gpu(0), source_array=imgb)
    img_itr = mx.io.NDArrayIter(data=img_nd, data_name='data',  batch_size=1)

    # get sym
    sym, args_params, aux_params = mx.model.load_checkpoint('drive_full_detect', 422)
    logit = sym.get_internals()['logit_output']
    mod = mx.mod.Module(symbol=logit, context=mx.gpu(0))
    mod.bind(img_itr.provide_data)
    mod.init_params(allow_missing=False, arg_params=args_params, aux_params=aux_params,
                    initializer=mx.init.Xavier(magnitude=2,rnd_type='gaussian',factor_type='in'))
    out = mod.predict(eval_data=img_itr, num_batch=2000)

    # show and record
    record = {}
    img_itr.reset()
    for i in range(2000):
        batch = img_itr.next()
        img = batch.data[0].asnumpy()[0].transpose((1,2,0))
        #label = batch.label[0].asnumpy().reshape((7,7,9))
        pred = (out.asnumpy()[i]+1)/2

        bboxlist = bboxdraw(img, pred)

        record[i] = bboxlist
        print ("Now is showing img {}".format(i))
    # write to json
    import json
    json = json.dumps(record)
    f = open(" testresult.json", "w")
    f.write(json)
    f.close()
