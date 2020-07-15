import pickle
import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

root_path = os.path.abspath(os.path.join('..'))
if root_path not in sys.path:
    sys.path.append(root_path)

import config_pretraining_ablation as config
from faces_loader import load_faces, build_dataset
from ssd_utils.ssd_loss import SSDLoss
from utils import import_by_name, train_test_split_tensors, MeanAveragePrecisionCallback

# Load train and validation data
train_image_paths, train_bnd_boxes = load_faces(split='train')
valid_image_paths, valid_bnd_boxes = load_faces(split='valid')
fake_image_paths, fake_bnd_boxes = load_faces(
    root=os.path.join('..', 'data', 'faces_fake'),
    split='fake')


valid_data = build_dataset(valid_image_paths, valid_bnd_boxes,
                           image_size=config.IMAGE_SIZE,
                           batch_size=config.BATCH_SIZE)


for run in range(1, config.NUM_RUNS+1):
    weights_dir = 'weights_pretraining_ablation_{}'.format(run)
    history_dir = 'history_pretraining_ablation_{}'.format(run)

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)

    for architecture in config.ARCHITECTURES:
        model_class = import_by_name('ssd_utils.networks.' + architecture)

        for prop_fake_samples in config.PROP_FAKE_SAMPLES:
            num_fake_samples = int(prop_fake_samples * config.NUM_REAL_SAMPLES)
            print('\n\nINFO: Considering {} real and {} fake samples'.format(
                config.NUM_REAL_SAMPLES, num_fake_samples))

            # Mixed training
            model_name = architecture.lower() + '_mixed'
            model_file = model_name + '_{}x.h5'.format(prop_fake_samples)
            model_path = os.path.join(weights_dir, model_file)

            if not os.path.exists(model_path):
                print('\n\nINFO: Training {} on mixed data..'.format(architecture))

                model = model_class(num_classes=len(config.CLASSES))
                anchors = model.get_anchors(image_shape=config.IMAGE_SIZE + (3,))

                real_data = build_dataset(train_image_paths[:config.NUM_REAL_SAMPLES],
                                          train_bnd_boxes[:config.NUM_REAL_SAMPLES],
                                          image_size=config.IMAGE_SIZE,
                                          batch_size=config.BATCH_SIZE,
                                          repeat=True, shuffle=True,
                                          encode_output=True,
                                          anchors=anchors,
                                          model=model)

                fake_data = build_dataset(fake_image_paths[:num_fake_samples],
                                          fake_bnd_boxes[:num_fake_samples],
                                          image_size=config.IMAGE_SIZE,
                                          batch_size=config.BATCH_SIZE,
                                          repeat=True, shuffle=True,
                                          encode_output=True,
                                          anchors=anchors,
                                          model=model)

                mixed_data = real_data.concatenate(fake_data)
                mixed_data = mixed_data.shuffle(100, reshuffle_each_iteration=True)

                meanAP_callback = MeanAveragePrecisionCallback(data=valid_data,
                                                               anchors=anchors)
                ckpt_callback = ModelCheckpoint(model_file,
                                                monitor='val_meanAP',
                                                mode='max',
                                                save_best_only=True,
                                                save_weights_only=True)

                model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                              loss=SSDLoss())

                history = model.fit(mixed_data, epochs=config.NUM_EPOCHS_MIXED,
                                    steps_per_epoch=config.STEPS_PER_EPOCH,
                                    callbacks=[meanAP_callback, ckpt_callback])

                history_file = model_name + '_history.pickle'
                history_path = os.path.join(history_dir, history_file)

                with open(history_path, 'wb') as f:
                    pickle.dump(history.history, f)

                model.load_weights(model_file)
                model.save_weights(model_path)
                os.remove(model_file)



            # Synthesized data pretraining
            model_name = architecture.lower() + '_pretrained'
            model_file = model_name + '_{}x.h5'.format(prop_fake_samples)
            model_path = os.path.join(weights_dir, model_file)

            pretrained_weights_file = model_path

            if not os.path.exists(model_path):
                print('\n\nINFO: Pretraining {} on fake data..'.format(architecture))

                model = model_class(num_classes=len(config.CLASSES))
                anchors = model.get_anchors(image_shape=config.IMAGE_SIZE + (3,))

                fake_data = build_dataset(fake_image_paths[:num_fake_samples],
                                          fake_bnd_boxes[:num_fake_samples],
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

                history = model.fit(fake_data, epochs=config.NUM_EPOCHS,
                                    steps_per_epoch=config.STEPS_PER_EPOCH,
                                    callbacks=[meanAP_callback, ckpt_callback])

                history_file = model_name + '_history.pickle'
                history_path = os.path.join(history_dir, history_file)

                with open(history_path, 'wb') as f:
                    pickle.dump(history.history, f)

                model.load_weights(model_file)
                model.save_weights(model_path)
                os.remove(model_file)


            # Finetuning
            model_name = architecture.lower() + '_finetuned'
            model_file = model_name + '_{}x.h5'.format(prop_fake_samples)
            model_path = os.path.join(weights_dir, model_file)

            if not os.path.exists(model_path):
                print('\n\nINFO: Finetuning {} on real data..'.format(architecture))

                model = model_class(num_classes=len(config.CLASSES))
                anchors = model.get_anchors(image_shape=config.IMAGE_SIZE + (3,))

                real_data = build_dataset(train_image_paths[:config.NUM_REAL_SAMPLES],
                                          train_bnd_boxes[:config.NUM_REAL_SAMPLES],
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


                print('INFO: Loading weights from %s'%pretrained_weights_file)
                model.load_weights(pretrained_weights_file)
                
                history = model.fit(real_data, epochs=config.NUM_EPOCHS,
                                    steps_per_epoch=config.STEPS_PER_EPOCH,
                                    callbacks=[meanAP_callback, ckpt_callback])

                history_file = model_name + '_history.pickle'
                history_path = os.path.join(history_dir, history_file)

                with open(history_path, 'wb') as f:
                    pickle.dump(history.history, f)
                
                model.load_weights(model_file)
                model.save_weights(model_path)
                os.remove(model_file)
