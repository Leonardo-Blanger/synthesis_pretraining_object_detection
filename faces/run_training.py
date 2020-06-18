import pickle
import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

root_path = os.path.abspath(os.path.join('..'))
if root_path not in sys.path:
    sys.path.append(root_path)

import config
from faces_loader import load_faces, build_dataset
from ssd_utils.ssd_loss import SSDLoss
from utils import import_by_name, train_test_split_tensors, MeanAveragePrecisionCallback

# Load train and validation data
train_image_paths, train_bnd_boxes = load_faces(split='train')
valid_image_paths, valid_bnd_boxes = load_faces(split='valid')

valid_data = build_dataset(valid_image_paths, valid_bnd_boxes,
                           image_size=config.IMAGE_SIZE,
                           batch_size=config.BATCH_SIZE)

for run in range(1, config.NUM_RUNS+1):
    weights_dir = 'weights_{}'.format(run)
    history_dir = 'history_{}'.format(run)

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)

    for architecture in config.ARCHITECTURES:
        model_class = import_by_name('ssd_utils.networks.' + architecture)

        model_name = architecture.lower() + '_pretrained'
        model_file = model_name + '.h5'
        model_path = os.path.join(weights_dir, model_file)
        pretrained_model_path = model_path

        if not os.path.exists(model_path):
            print('\n\nINFO: Pretraining {} on fake data..'.format(architecture))

            model = model_class(num_classes=len(config.CLASSES))
            anchors = model.get_anchors(image_shape=config.IMAGE_SIZE + (3,))

            fake_image_paths, fake_bnd_boxes = load_faces(
                root=os.path.join('..', 'data', 'faces_fake'),
                split='fake')

            fake_data = build_dataset(fake_image_paths, fake_bnd_boxes,
                                      image_size=config.IMAGE_SIZE,
                                      batch_size=config.BATCH_SIZE,
                                      repeat=True, shuffle=True,
                                      encode_output=True,
                                      anchors=anchors,
                                      model=model)

            meanAP_callback = MeanAveragePrecisionCallback(data=valid_data,
                                                           anchors=anchors)
            ckpt_callback = ModelCheckpoint(model_file,
                                            monitor='val_meanAP',
                                            mode='max',
                                            save_best_only=True,
                                            save_weights_only=True)

            model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                          loss=SSDLoss())

            history = model.fit(fake_data, epochs=config.NUM_EPOCHS_PRETRAIN,
                                steps_per_epoch=config.STEPS_PER_EPOCH,
                                callbacks=[meanAP_callback, ckpt_callback])

            history_file = model_name + '_history.pickle'
            history_path = os.path.join(history_dir, history_file)

            with open(history_path, 'wb') as f:
                pickle.dump(history.history, f)

            model.load_weights(model_file)
            model.save_weights(model_path)
            os.remove(model_file)
            del model, fake_data

        for train_samples in config.TRAIN_SAMPLES:
            (_, small_train_image_paths,
             _, small_train_bnd_boxes) = train_test_split_tensors(
                 train_image_paths, train_bnd_boxes, test_size=train_samples,
                 random_state=config.RANDOM_SEED)

            assert small_train_image_paths.shape[0] == small_train_bnd_boxes.shape[0]
            assert small_train_image_paths.shape[0] == train_samples

            for train_type in ['from_scratch', 'finetuned']:
                model_name = architecture.lower() + '_{}_samples_{}'.format(
                    train_samples, train_type)
                model_file = model_name + '.h5'
                model_path = os.path.join(weights_dir, model_file)

                if not os.path.exists(model_path):
                    print('\n\nINFO: Training {} {} on {} samples'.format(
                        architecture, train_type, train_samples))

                    model = model_class(num_classes=len(config.CLASSES))
                    anchors = model.get_anchors(image_shape=config.IMAGE_SIZE + (3,))

                    train_data = build_dataset(small_train_image_paths,
                                               small_train_bnd_boxes,
                                               image_size=config.IMAGE_SIZE,
                                               batch_size=config.BATCH_SIZE,
                                               repeat=True, shuffle=True,
                                               encode_output=True,
                                               anchors=anchors,
                                               model=model)

                    if train_type == 'finetuned':
                        model.build(input_shape=(None,) + config.IMAGE_SIZE + (3,))
                        model.load_weights(pretrained_model_path)

                    meanAP_callback = MeanAveragePrecisionCallback(data=valid_data,
                                                                   anchors=anchors)
                    ckpt_callback = ModelCheckpoint(model_file,
                                                    monitor='val_meanAP',
                                                    mode='max',
                                                    save_best_only=True,
                                                    save_weights_only=True)

                    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                                  loss=SSDLoss())

                    history = model.fit(train_data, epochs=config.NUM_EPOCHS,
                                        steps_per_epoch=config.STEPS_PER_EPOCH,
                                        callbacks=[meanAP_callback, ckpt_callback])

                    history_file = model_name + '_history.pickle'
                    history_path = os.path.join(history_dir, history_file)

                    with open(history_path, 'wb') as f:
                        pickle.dump(history.history, f)

                    model.load_weights(model_file)
                    model.save_weights(model_path)
                    os.remove(model_file)
                    del model, train_data
