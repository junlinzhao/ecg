import argparse
import json
import keras
import numpy as np
import os
import random
import time

import network
import load
import util

MAX_EPOCHS = 100

def make_save_dir(dirname, experiment_name):
    start_time = str(int(time.time())) + '-' + str(random.randrange(1000))
    save_dir = os.path.join(dirname, experiment_name, start_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def get_filename_for_saving(save_dir):
    return os.path.join(save_dir,
            "{val_loss:.3f}-{val_acc:.3f}-{epoch:03d}-{loss:.3f}-{acc:.3f}.hdf5")

def train(args, params):

    print("Loading training set...")
    train = load.load_dataset(params['train'])
    print("Loading dev set...")
    dev = load.load_dataset(params['dev'])
    print("Building preprocessor...")
    preproc = load.Preproc(*train)
    print("Training size: " + str(len(train[0])) + " examples.")
    print("Dev size: " + str(len(dev[0])) + " examples.")


    save_dir = make_save_dir(params['save_dir'], args.experiment)

    util.save(preproc, save_dir)

    params.update({
        "input_shape": [None, 1],
        "num_categories": len(preproc.classes)
    })
    '''
    <class 'dict'>: 
    {'conv_subsample_lengths': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2], 
    'conv_filter_length': 16, 
    'conv_num_filters_start': 32,
     'conv_init': 'he_normal', 
     'conv_activation': 'relu', 
     'conv_dropout': 0.2, 
     'conv_num_skip': 2, 
     'conv_increase_channels_at': 4, 
     'learning_rate': 0.001, 
     'batch_size': 32, 
     'train': '../examples/cinc17/train.json', 
     'dev': '../examples/cinc17/dev.json', 
     'generator': True, 
     'save_dir': 'saved', 
     'input_shape': [None, 1], 
     'num_categories': 4}
    '''
    model = network.build_network(**params)

    stopping = keras.callbacks.EarlyStopping(patience=8)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=0.1,
        patience=2,
        min_lr=params["learning_rate"] * 0.001)

    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=get_filename_for_saving(save_dir),
        save_best_only=False)

    batch_size = params.get("batch_size", 32)

    #把网络结构打出来
    # model.summary()


    if params.get("generator", False):
        train_gen = load.data_generator(batch_size, preproc, *train)
        dev_gen = load.data_generator(batch_size, preproc, *dev)
        model.fit_generator(
            train_gen,
            steps_per_epoch=int(len(train[0]) / batch_size),
            epochs=MAX_EPOCHS,
            validation_data=dev_gen,
            validation_steps=int(len(dev[0]) / batch_size),
            callbacks=[checkpointer, reduce_lr, stopping])
    else:
        train_x, train_y = preproc.process(*train)
        dev_x, dev_y = preproc.process(*dev)
        model.fit(
            train_x, train_y,
            batch_size=batch_size,
            epochs=MAX_EPOCHS,
            validation_data=(dev_x, dev_y),
            callbacks=[checkpointer, reduce_lr, stopping])
    #保存训练结果
    model.save('my_model.h5')
    #返回训练之后的模型
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("--experiment", "-e", help="tag with experiment name",
                        default="default")
    args = parser.parse_args()

    # print(args)
    # args=object()
    args.config_file="../examples/cinc17/config.json"
    args.experiment='cinc17'
    params = json.load(open(args.config_file, 'r'))
    print(params)
    params["train"]="../"+ params["train"]
    params["dev"] = "../" + params["dev"]
    model= train(args, params)

