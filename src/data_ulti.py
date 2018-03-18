import json

import cv2
import mxnet as mx
import numpy as np

IMGROOT = "./DATA/"


# get iterator
def get_iterator(path, data_shape, label_width, batch_size, shuffle=False):
    iterator = mx.io.ImageRecordIter(path_imgrec=path,
                                     data_shape=data_shape,
                                     label_width=label_width,
                                     batch_size=batch_size,
                                     shuffle=shuffle)
    return iterator


# Convert data to rec file
def get_YOLO_xy(bxy, grid_size=(7, 7), dscale=32, sizet=224):
    cx, cy = bxy
    assert cx <= 1 and cy <= 1, "All should be < 1, but get {}, and {}".format(cx, cy)

    j = int(np.floor(cx / (1.0 / grid_size[0])))
    i = int(np.floor(cy / (1.0 / grid_size[1])))
    xyolo = (cx * sizet - j * dscale) / dscale
    yyolo = (cy * sizet - i * dscale) / dscale
    return [i, j, xyolo, yyolo]


# Get YOLO label
def imgResizeBBoxTransform(img, bbox, sizet, grid_size=(7, 7, 5), dscale=32):
    himg, wimg = img.shape[:2]
    imgR = cv2.resize(img, dsize=(sizet, sizet))
    bboxyolo = np.zeros(grid_size)
    for eachbox in bbox:
        cx, cy, w, h = eachbox[:4]
        cls = eachbox[4:]
        cxt = 1.0 * cx / wimg
        cyt = 1.0 * cy / himg
        wt = 1.0 * w / wimg
        ht = 1.0 * h / himg
        assert wt < 1 and ht < 1
        i, j, xyolo, yyolo = get_YOLO_xy([cxt, cyt], grid_size, dscale, sizet)
        # print "one yolo box is {}".format((i, j, xyolo, yyolo, wt, ht))
        label_vec = np.asarray([1, xyolo, yyolo, wt, ht] + cls)
        # print "Final yolo box is {}".format(label_vec)
        bboxyolo[i, j, :] = label_vec
    return imgR, bboxyolo


# Convert raw images to rec files
def toRecFile(imgroot, imglist, annotation, sizet, grid_size, dscale):
    record = mx.recordio.MXIndexedRecordIO("drive_full.idx",
                                           "drive_full.rec", 'w')
    for i in range(len(imglist)):
        imgname = imglist[i]
        img = cv2.imread(imgroot + imgname)
        bbox = annotation[imgname]
        print ("Now is processing img {}".format(imgname))
        imgR, bboxR = imgResizeBBoxTransform(img, bbox, sizet, grid_size, dscale)
        header = mx.recordio.IRHeader(flag=0, label=bboxR.flatten(), id=0, id2=0)
        s = mx.recordio.pack_img(header, imgR, quality=100, img_fmt='.jpg')
        record.write_idx(i, s)
    print ("JPG to rec is Done")
    record.close()


def idltonumpy(idlfile, savepath):
    label_np = {}
    with open(idlfile) as f:
        for line in f:
            jsonload = json.loads(line)
            assert len(jsonload.keys()) == 1, "Only one image per json file"
            label_np[jsonload.keys()[0]] = jsonload[jsonload.keys()[0]]
    np.save(save_path, label_np)
    return label_np


def xy2wh(idlnpy):
    labelwh = {}
    for key in idlnpy.keys():
        boxxy = idlnpy[key]
        boxeswh = []
        for each in boxxy:
            x1, y1, x2, y2, idx = each
            cx = int((x2 + x1) / 2)
            cy = int((y1 + y2) / 2)
            w = int(x2 - x1)
            h = int(y2 - y1)
            cls = [0, 0, 0, 0]
            if idx == 1:
                cls[0] = 1
            elif idx == 2:
                cls[1] = 1
            elif idx == 3:
                cls[2] = 1
            elif idx == 20:
                cls[3] = 1
            else:
                print (idx)
                print ("Not expected value")
                break
            assert w > 0 and h > 0, "wh should be > 0"
            box = [cx, cy, w, h] + cls
            boxeswh.append(box)
        labelwh[key] = boxeswh
    return labelwh


if __name__ == "__main__":
    # transform jpg to rec file
    labelfile = "./DATA/label.idl"
    save_path = "./DATA/label.npy"
    idlnpy = idltonumpy(labelfile, save_path)
    labelwh = xy2wh(idlnpy)
    imglist = labelwh.keys()
    sizet = 224
    toRecFile(IMGROOT, imglist, labelwh, sizet, (7, 7, 9), 32)
