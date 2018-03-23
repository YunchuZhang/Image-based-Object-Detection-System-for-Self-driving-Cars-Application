import json
import cv2
import numpy as np

if __name__ == "__main__":
    #read label
    with open("../DATA/label.idl", "r")as f:
        lines = f.readlines()
    label = {}
    for line in lines:
        l = json.loads(line)
        label.update(l)
    #data agument
    record = {}
    imglist1 = []
    r1 = []
    r2 = []
    cnt = 1
    #10001
    #agu 1
    for i in range(10001):
        imglist1.append("{}".format(60090 + i) + ".jpg")

    for i in range(1, len(imglist1)):
        img = cv2.imread("../DATA/" + imglist1[i])
        # mid = 320
        crop = 330
        hi, wi = img.shape[:2]
        crop1 = img[:hi, :crop]
        crop2 = img[:hi, crop + 1:-1]
        print('now is cut' + '%d' % (i) + '.jpg')
        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)


        cv2.imwrite('%d' % (2 * i - 1) + '.jpg', crop1);
        cv2.imwrite('%d' % (2 * i) + '.jpg', crop2);

        # giveorigin lable
        a = label[imglist1[i]]
        a = np.array(a)
        if (a.shape[0]) > 0:
            for j in range(a.shape[0]):

                if (a[j][2] <= crop):

                    r1.append([a[j][0], a[j][1], a[j][2], a[j][3], a[j][4]])
                    # np.float64(bboxl).tolist()

                elif (a[j][0] >= crop):
                    r2.append([a[j][0] - crop, a[j][1], a[j][2] - crop, a[j][3], a[j][4]])

                else:

                    r1.append([a[j][0], a[j][1], crop, a[j][3], a[j][4]])

                    r2.append([0, a[j][1], a[j][2] - crop, a[j][3], a[j][4]])

            record[2 * i - 1] = np.float64(r1).tolist()

            record[2 * i] = np.float64(r2).tolist()

        else:
            record[2 * i - 1] = np.float64(r1).tolist()

            record[2 * i] = np.float64(r2).tolist()
        del r1[:]
        del r2[:]
    # agu 2
    for i in range(1, len(imglist1)):
        img = cv2.imread("../DATA/" + imglist1[i])
        # mid = 320
        crop = 275
        hi, wi = img.shape[:2]
        crop1 = img[:hi, :crop]
        crop2 = img[:hi, crop + 1:-1]
        print('now is cut' + '%d' % (i) + '.jpg')
        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)
        k = i+len(imglist1)-1

        cv2.imwrite('%d' % (2 * k - 1) + '.jpg', crop1);
        cv2.imwrite('%d' % (2 * k) + '.jpg', crop2);

        # giveorigin lable
        a = label[imglist1[i]]
        a = np.array(a)
        if (a.shape[0]) > 0:
            for j in range(a.shape[0]):

                if (a[j][2] <= crop):

                    r1.append([a[j][0], a[j][1], a[j][2], a[j][3], a[j][4]])
                    # np.float64(bboxl).tolist()

                elif (a[j][0] >= crop):
                    r2.append([a[j][0] - crop, a[j][1], a[j][2] - crop, a[j][3], a[j][4]])

                else:

                    r1.append([a[j][0], a[j][1], crop, a[j][3], a[j][4]])

                    r2.append([0, a[j][1], a[j][2] - crop, a[j][3], a[j][4]])

            record[2 * k - 1] = np.float64(r1).tolist()

            record[2 * k] = np.float64(r2).tolist()

        else:
            record[2 * k - 1] = np.float64(r1).tolist()

            record[2 * k] = np.float64(r2).tolist()
        del r1[:]
        del r2[:]

    # add origin
    for i in range(1, len(imglist1)):
        img = cv2.imread("../DATA/" + imglist1[i])
        k = 4*(len(imglist1)-1)
        cv2.imwrite('%d' % (k+i) + '.jpg', img);
        print('now is origin' + '%d' % (i) + '.jpg')

        # giveorigin lable
        a = label[imglist1[i]]
        a = np.array(a)
        if (a.shape[0]) > 0:
            for j in range(a.shape[0]):
                r1.append([a[j][0], a[j][1], a[j][2], a[j][3], a[j][4]])

            record[k+i] = np.float64(r1).tolist()

        else:
            record[k+i] = np.float64(r1).tolist()

        del r1[:]

    import json

    json = json.dumps(record)
    f = open("testresult.json", "w")
    f.write(json)
    f.close()