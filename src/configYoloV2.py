from easydict import EasyDict as edict

# info
info = edict()
info.name = "network1_Yolov2"

# training setting
train = edict()

train.num_gpu = 16
train.base_lr = 0.0002 * train.num_gpu
train.stop_factor_lr = 1e-4
train.drop_factor = 0.1
train.drop_step = 100000
train.optimizer = 'adam'
train.epoch = 300
train.save_path = "OCR_FoI_Training/model/networkYoloV2_contest_data"
train.prob_scale = 1
train.object_scale = 1
train.noobject_scale = 0.1
train.coord_scale = 5
train.anchors_num = 5
train.anchors = [[0.10235532, 0.04075217],
                 [0.29644471, 0.05406184],
                 [0.04517153, 0.02728633],
                 [0.1752567, 0.0486942],
                 [0.55452996, 0.06665693]]

# network settings
network = edict()

network.resnet = edict()
network.resnet.pretrained = ''
network.resnet.pretrained_epoch = 0
network.resnet.Dilate = 16
network.resnet.num_layers = 34
network.resnet.num_classes = 10  # doesnt matter whatever this number is
network.resnet.conv_workspace = 512  # MB
network.resnet.layer_extract = "last_conv"  # "_plus12_output" #"_plus5_output"#
network.resnet.image_shape = "3,512,512"
network.resnet.network_name = "OCR_FoI_Training.symbol.resnet_xt"
# network.resnet.network_name = "OCR_FoI_Training.symbol.resnet_fcrn"
# network.resnet.network_name = "OCR_FoI_Training.symbol.resnext_fcrn"
network.resnet.out_size = (32, 32, 7 * train.anchors_num)
network.resnet.lossname = "YoloV2"

# dataset settings
dataset = edict()

dataset.data_train_path = "path/to/training_rec"
dataset.data_val_path = "path/to/validation_rec"
dataset.data_test_path = "path/to/testing_rec"
dataset.data_shape = (3, 512, 512)
dataset.label_width = (1024 * 7)
dataset.shuffle = True
dataset.batch_size = 9 * train.num_gpu

# fine-tune
tune = edict()
tune.if_tune = True
# edit to the model for parameters init
tune.tune_name = "OCR_FoI_Training/model/models/network1_final"
tune.tune_epoch = 31

# monitor grads
monitor = edict()
monitor.if_monitor = False
monitor.pattern = '.*linearregressionoutput*.'
monitor.stat = 'norm'
monitor.step = 2

# reconstruct loss function
recon = edict()
recon.if_recon = True


def generate_config(_network):
    config = {}

    for k, v in network[_network].items():
        config[k] = v
    for k, v in dataset.items():
        config[k] = v
    for k, v in train.items():
        config[k] = v
    for k, v in tune.items():
        config[k] = v
    for k, v in monitor.items():
        config[k] = v
    for k, v in recon.items():
        config[k] = v
    return config