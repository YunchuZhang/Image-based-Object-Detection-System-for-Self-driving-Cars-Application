import mxnet as mx 
from Symbol.symbol import get_resnet_model
import numpy as np
from data_ulti import get_iterator
from tools.logging_metric import LogMetricsCallback, LossMetric
import time

import logging
import sys
root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
root_logger.addHandler(stdout_handler)
root_logger.setLevel(logging.DEBUG)

if __name__ == "__main__":

    # get sym
    # Try different network 18, 50, 101 to find the best one
    sym = get_resnet_model('pretrained_models/resnet-34', 0)
    _, args_params, aux_params = mx.model.load_checkpoint('pretrained_models/resnet-34', 0)

    # get some input
    # change it to the data rec you create, and modify the batch_size
    train_data = get_iterator(path='DATA_rec/drive_full.rec', data_shape=(3, 224, 224), label_width=7*7*9, batch_size=32, shuffle=True)
    val_data = get_iterator(path='DATA_rec/drive_full.rec', data_shape=(3, 224, 224), label_width=7*7*9, batch_size=32)
    
    # allocate gpu/cpu mem to the sym
    mod = mx.mod.Module(symbol=sym, context=mx.gpu(0))

    # setup metric
    # metric = mx.metric.create(loss_metric, allow_extra_outputs=True)
    tme = time.time()
    logtrain = LogMetricsCallback('logs/train_'+str(tme))

    # setup monitor for debugging
    def norm_stat(d):
        return mx.nd.norm(d) / np.sqrt(d.size)
    mon = None #mx.mon.Monitor(10, norm_stat, pattern=".*backward*.")

    # save model
    checkpoint = mx.callback.do_checkpoint('save_model/drive_full_detect')

    # Train
    # Try different hyperparamters to get the model converged, (batch_size,
    # optimization method, training epoch, learning rate/scheduler)
    mod.fit(train_data=train_data,
            eval_data=val_data,
            num_epoch=600,
            monitor=mon,
            eval_metric=LossMetric(0.5),
            optimizer='rmsprop',
            optimizer_params={'learning_rate':0.01, 'lr_scheduler': mx.lr_scheduler.FactorScheduler(300000, 0.1, 0.001)},
            initializer=mx.init.Xavier(magnitude=2, rnd_type='gaussian', factor_type='in'),
            arg_params=args_params,
            aux_params=aux_params,
            allow_missing=True,
            batch_end_callback=[mx.callback.Speedometer(batch_size=32, frequent=10, auto_reset=False), logtrain],
            epoch_end_callback=checkpoint,
             )
