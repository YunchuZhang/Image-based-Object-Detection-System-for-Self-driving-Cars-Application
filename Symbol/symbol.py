import unittest

import mxnet as mx
import numpy as np


@mx.init.register
class CustomConstant(mx.init.Initializer):
    def __init__(self, value):
        super(CustomConstant, self).__init__(value=value)
        self.value = value

    def _init_weight(self, _, arr):
        arr[:] = mx.nd.array(self.value)


class ConstantInitTest(unittest.TestCase):
    def testInitCPU(self, ):
        batch_size = 10

        const_arr = (np.ones((5, 5)) * 10).tolist()
        a = mx.sym.Variable('a', shape=(5, 5), init=CustomConstant(value=const_arr))
        a = mx.sym.BlockGrad(a)  # now variable a is a constant

        data = mx.sym.Variable('data')
        loss = mx.sym.MakeLoss(mx.sym.broadcast_add(a * (-1), data) ** 2)

        out = mx.sym.Group([loss, mx.sym.BlockGrad(a)])
        mod = mx.mod.Module(out, data_names=['data'], label_names=[])
        mod.bind(data_shapes=[('data', (batch_size, 5, 5)), ])
        mod.init_params(initializer=mx.init.Uniform())
        mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.005),))

        # a = mx.nd.ones((5, 5))
        data = np.ones((1000, 5, 5))

        dataiter = mx.io.NDArrayIter(data={'data': data}, batch_size=batch_size)
        dataiter.reset()
        for batch_id, databatch in enumerate(dataiter):
            mod.forward_backward(databatch)
            mod.update()
            assert (mod.get_outputs()[1].asnumpy() == const_arr).all()


def expit_tensor(x):
    return 1/(1+mx.sym.exp(-x))


# Yolo loss
def YOLO_loss(predict, label):
    """
    predict (params): mx.sym->which is NDarray (tensor), its shape is (batch_size, 7, 7,5 )
    label: same as predict
    """
    # Reshape input to desired shape
    predict = mx.sym.reshape(predict, shape=(-1, 49, 9))
    # shift everything to (0, 1)
    predict_shift = (predict+1)/2
    label = mx.sym.reshape(label, shape=(-1, 49, 9))
    # split the tensor in the order of [prob, x, y, w, h]
    cl, xl, yl, wl, hl, clsl1, clsl2, clsl3, clsl4 = mx.sym.split(label, num_outputs=9, axis=2)
    cp, xp, yp, wp, hp, clsp1, clsp2, clsp3, clsp4 = mx.sym.split(predict_shift, num_outputs=9, axis=2)
    # clsesl = mx.sym.Concat(clsl1, clsl2, clsl3, clsl4, dim=2)
    # clsesp = mx.sym.Concat(clsp1, clsp2, clsp3, clsp4, dim=2)
    # weight different target differently
    lambda_coord = 5
    lambda_obj = 1
    lambda_noobj = 0.2
    mask = cl*lambda_obj+(1-cl)*lambda_noobj

    # linear regression
    lossc = mx.sym.LinearRegressionOutput(label=cl*mask, data=cp*mask)
    lossx = mx.sym.LinearRegressionOutput(label=xl*cl*lambda_coord, data=xp*cl*lambda_coord)
    lossy = mx.sym.LinearRegressionOutput(label=yl*cl*lambda_coord, data=yp*cl*lambda_coord)
    lossw = mx.sym.LinearRegressionOutput(label=mx.sym.sqrt(wl)*cl*lambda_coord, data=mx.sym.sqrt(wp)*cl*lambda_coord)
    lossh = mx.sym.LinearRegressionOutput(label=mx.sym.sqrt(hl)*cl*lambda_coord, data=mx.sym.sqrt(hp)*cl*lambda_coord)
    losscls1 = mx.sym.LinearRegressionOutput(label=clsl1*cl, data=clsp1*cl)
    losscls2 = mx.sym.LinearRegressionOutput(label=clsl2*cl, data=clsp2*cl)
    losscls3 = mx.sym.LinearRegressionOutput(label=clsl3*cl, data=clsp3*cl)
    losscls4 = mx.sym.LinearRegressionOutput(label=clsl4*cl, data=clsp4*cl)
    losscls = losscls1+losscls2+losscls3+losscls4
    # return joint loss
    loss = lossc+lossx+lossy+lossw+lossh+losscls
    return loss

def get_resnet_model(model_path, epoch):
    label = mx.sym.Variable('softmax_label')
    sym,args,aux = mx.model.load_checkpoint(model_path,epoch)
    sym = sym.get_internals()['bn1_output']
    sym = mx.sym.Activation(data = sym, act_type = "relu")
    sym = mx.sym.Convolution(data = sym, kernel=(3,3), num_filter = 9, pad=(1,1),
                             stride= (1,1), no_bias=True)

    sym = sym/(1+mx.sym.abs(sym))
    logit = mx.sym.transpose(sym, axes=(0, 2, 3, 1), name="logit") # (-1, 7, 7, 5(c,x,y,w,h))
    # apply loss
    loss_ = YOLO_loss(logit, label)
    # mxnet special requirement
    loss = mx.sym.MakeLoss(loss_)
    # multi-output logit should be blocked from generating gradients
    out = mx.sym.Group([mx.sym.BlockGrad(logit), loss])
    return out

def get_symbol_train_YoloV2_2(model_path, epoch):

    # setup label
    label = mx.sym.Variable("softmax")


    # setup the anchors
    num_anchor = 5
    anchors_num = [[0.10235532, 0.04075217],
                    [0.29644471, 0.05406184],
                    [0.04517153, 0.02728633],
                    [0.1752567, 0.0486942],
                    [0.55452996, 0.06665693]]
    anchors = mx.sym.Variable("anchors", shape=(5, 2), dtype=np.float32,
                              init=CustomConstant(anchors_num))
    anchors = mx.sym.BlockGrad(anchors)
    anchors_w, anchors_h = mx.sym.split(anchors, axis=1, num_outputs=2, name="anchor_split")

    # read resnet
    resnet, args, aux = mx.model.load_checkpoint(model_path, epoch)
    resnet_interals = resnet.get_internals()['bn1_output']

    # objectness + classification
    conv_obj = mx.sym.Convolution(data=resnet_interals, kernel=(3, 3), pad=(1, 1),
                                  num_filter=5, name="obj_cls")
    conv_coord = mx.sym.Convolution(data=resnet_interals, kernel=(3,3), pad=(1, 1),
                                    num_filter=5*num_anchor, name="coord_anchor")

    # build up loss layer
    # object + classification
    cp, cls1, cls2, cls3, cls4 = mx.sym.split(conv_obj, num_outputs=5, axis=3)
    prob_adjust = expit_tensor(cp)
    cls1p = expit_tensor(cls1)
    cls2p = expit_tensor(cls2)
    cls3p = expit_tensor(cls3)
    cls4p = expit_tensor(cls4)


    # bbox regression
    b1, x1, y1, w1, h1, \
    b2, x2, y2, w2, h2, \
    b3, x3, y3, w3, h3, \
    b4, x4, y4, w4, h4, \
    b5, x5, y5, w5, h5, = mx.sym.split(conv_coord, num_outputs=25, axis=3)

    b_all = mx.sym.stack(b1, b2, b3, b4, b5, axis=3)
    x_all = mx.sym.stack(x1, x2, x3, x4, x5, axis=3)
    y_all = mx.sym.stack(y1, y2, y3, y4, y5, axis=3)
    w_all = mx.sym.stack(w1, w2, w3, w4, w5, axis=3)
    h_all = mx.sym.stack(h1, h2, h3, h4, h5, axis=3)

    bprob = expit_tensor(b_all)
    x_adjust = expit_tensor(x_all) - 0.5  # sigmoid and shift it to -0.5~0.5
    y_adjust = expit_tensor(y_all) - 0.5  # sigmoid and shift it to -0.5~0.5
    w_adjust = mx.sym.sqrt(
        mx.sym.broadcast_mul(mx.sym.exp(w_all), mx.sym.reshape(anchors_w, shape=[1, 1, 1, num_anchor])))
    h_adjust = mx.sym.sqrt(
        mx.sym.broadcast_mul(mx.sym.exp(h_all), mx.sym.reshape(anchors_h, shape=[1, 1, 1, num_anchor])))


    # todo: either do translation on GT, or do the mapping here.
    # prob_adjust cls1p cls2p cls3p cls4p
    # bprob, x_adjust, y_adjust, w_adujst, h_adjust







