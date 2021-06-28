import time

import keras
from keras.layers import Conv2D, BatchNormalization, ELU, Permute
from classifiers.classifiers import predict_model
from utils.tools import save_logs



# Most of the code here are taken from https://github.com/hfawaz/dl-4-tsc

class Classifier_D_ResNet:
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False):
        if verbose:
            print('[D_ResNet] Creating Disjoint ResNet Classifier')
        self.verbose = verbose
        self.output_directory = output_directory
        self.model = self.build_model(input_shape, nb_classes)
        if verbose:
            self.model.summary()

        self.model.save_weights(self.output_directory + 'model_init.h5')

    def build_model(self, input_shape, nb_classes):

        n_feature_maps = 32
        input_layer = keras.layers.Input(input_shape)
        # BLOCK 1
        # Temporal Convolutions
        conv1_x = Conv2D(n_feature_maps, (8, 1), strides=1, padding="same")(input_layer)
        conv1_x = BatchNormalization()(conv1_x)
        conv1_x = ELU(alpha=1.0)(conv1_x)
        # Spatial Convolutions
        conv1_x = Conv2D(n_feature_maps, (1, input_shape[1]), strides=1, padding="valid")(conv1_x)
        conv1_x = BatchNormalization()(conv1_x)
        conv1_x = ELU(alpha=1.0)(conv1_x)
        conv1_x = Permute((1, 3, 2))(conv1_x)

        # Temporal Convolutions
        conv1_y = Conv2D(n_feature_maps, (5, 1), strides=1, padding="same")(conv1_x)
        conv1_y = BatchNormalization()(conv1_y)
        conv1_y = ELU(alpha=1.0)(conv1_y)
        # Spatial Convolutions
        conv1_y = Conv2D(n_feature_maps, (1, n_feature_maps), strides=1, padding="valid")(conv1_y)
        conv1_y = BatchNormalization()(conv1_y)
        conv1_y = ELU(alpha=1.0)(conv1_y)
        conv1_y = Permute((1, 3, 2))(conv1_y)

        # Temporal Convolutions
        conv1_z = Conv2D(n_feature_maps, (3, 1), strides=1, padding="same")(conv1_y)
        conv1_z = BatchNormalization()(conv1_z)
        conv1_z = ELU(alpha=1.0)(conv1_z)
        # Spatial Convolutions
        conv1_z = Conv2D(n_feature_maps, (1, n_feature_maps), strides=1, padding="valid")(conv1_z)
        conv1_z = BatchNormalization()(conv1_z)
        conv1_z = Permute((1, 3, 2))(conv1_z)

        # expand channels for the sum
        # Temporal Convolutions
        shortcut_y = Conv2D(n_feature_maps, (1, 1), strides=1, padding="same")(input_layer)
        shortcut_y = BatchNormalization()(shortcut_y)
        shortcut_y = ELU(alpha=1.0)(shortcut_y)
        # Spatial Convolutions
        shortcut_y = Conv2D(n_feature_maps, (1, input_shape[1]), strides=1, padding="valid")(shortcut_y)
        shortcut_y = BatchNormalization()(shortcut_y)
        shortcut_y = Permute((1, 3, 2))(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv1_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2
        # Temporal Convolutions
        conv2_x = Conv2D(n_feature_maps, (8, 1), strides=1, padding="same")(output_block_1)
        conv2_x = BatchNormalization()(conv2_x)
        conv2_x = ELU(alpha=1.0)(conv2_x)
        # Spatial Convolutions
        conv2_x = Conv2D(n_feature_maps * 2, (1, n_feature_maps), strides=1, padding="valid")(conv2_x)
        conv2_x = BatchNormalization()(conv2_x)
        conv2_x = ELU(alpha=1.0)(conv2_x)
        conv2_x = Permute((1, 3, 2))(conv2_x)

        # Temporal Convolutions
        conv2_y = Conv2D(n_feature_maps, (5, 1), strides=1, padding="same")(conv2_x)
        conv2_y = BatchNormalization()(conv2_y)
        conv2_y = ELU(alpha=1.0)(conv2_y)
        # Spatial Convolutions
        conv2_y = Conv2D(n_feature_maps*2, (1, n_feature_maps*2), strides=1, padding="valid")(conv2_y)
        conv2_y = BatchNormalization()(conv2_y)
        conv2_y = ELU(alpha=1.0)(conv2_y)
        conv2_y = Permute((1, 3, 2))(conv2_y)

        # Temporal Convolutions
        conv2_z = Conv2D(n_feature_maps, (3, 1), strides=1, padding="same")(conv2_y)
        conv2_z = BatchNormalization()(conv2_z)
        conv2_z = ELU(alpha=1.0)(conv2_z)
        # Spatial Convolutions
        conv2_z = Conv2D(n_feature_maps * 2, (1, n_feature_maps*2), strides=1, padding="valid")(conv2_z)
        conv2_z = BatchNormalization()(conv2_z)
        conv2_z = Permute((1, 3, 2))(conv2_z)

        # expand channels for the sum
        # Temporal Convolutions
        shortcut2_y = Conv2D(n_feature_maps, (1, 1), strides=1, padding="same")(output_block_1)
        shortcut2_y = BatchNormalization()(shortcut2_y)
        shortcut2_y = ELU(alpha=1.0)(shortcut2_y)
        # Spatial Convolutions
        shortcut2_y = Conv2D(n_feature_maps * 2, (1, n_feature_maps), strides=1, padding="valid")(shortcut2_y)
        shortcut2_y = BatchNormalization()(shortcut2_y)
        shortcut2_y = Permute((1, 3, 2))(shortcut2_y)

        output_block_2 = keras.layers.add([shortcut2_y, conv2_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3
        # Temporal Convolutions
        conv3_x = Conv2D(n_feature_maps, (8, 1), strides=1, padding="same")(output_block_2)
        conv3_x = BatchNormalization()(conv3_x)
        conv3_x = ELU(alpha=1.0)(conv3_x)
        # Spatial Convolutions
        conv3_x = Conv2D(n_feature_maps * 2, (1, n_feature_maps*2), strides=1, padding="valid")(conv3_x)
        conv3_x = BatchNormalization()(conv3_x)
        conv3_x = ELU(alpha=1.0)(conv3_x)
        conv3_x = Permute((1, 3, 2))(conv3_x)

        # Temporal Convolutions
        conv3_y = Conv2D(n_feature_maps, (5, 1), strides=1, padding="same")(conv3_x)
        conv3_y = BatchNormalization()(conv3_y)
        conv3_y = ELU(alpha=1.0)(conv3_y)
        # Spatial Convolutions
        conv3_y = Conv2D(n_feature_maps * 2, (1, n_feature_maps * 2), strides=1, padding="valid")(conv3_y)
        conv3_y = BatchNormalization()(conv3_y)
        conv3_y = ELU(alpha=1.0)(conv3_y)
        conv3_y = Permute((1, 3, 2))(conv3_y)

        # Temporal Convolutions
        conv3_z = Conv2D(n_feature_maps, (3, 1), strides=1, padding="same")(conv3_y)
        conv3_z = BatchNormalization()(conv3_z)
        conv3_z = ELU(alpha=1.0)(conv3_z)
        # Spatial Convolutions
        conv3_z = Conv2D(n_feature_maps * 2, (1, n_feature_maps * 2), strides=1, padding="valid")(conv3_z)
        conv3_z = BatchNormalization()(conv3_z)
        conv3_z = Permute((1, 3, 2))(conv3_z)

        # no need to expand channels because they are equal
        shortcut3_y = keras.layers.normalization.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut3_y, conv3_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)
        output_block_3 = Permute((1, 3, 2))(output_block_3)

        # FINAL
        gap_layer = keras.layers.GlobalAveragePooling2D()(output_block_3)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        return model

    def fit(self, Ximg_train, yimg_train, Ximg_val=None, yimg_val=None, epochs=10, batch_size=16):
        if self.verbose:
            print('[D_ResNet] Training Disjoint ResNet Classifier')
        mini_batch_size = int(min(Ximg_train.shape[0] / 10, batch_size))

        self.model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

        file_path = self.output_directory + 'best_model.h5'
        if Ximg_val is not None:
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                          min_lr=0.0001)
            model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                                               save_best_only=True)
        else:
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                          min_lr=0.0001)
            model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                               save_best_only=True)
        self.callbacks = [reduce_lr, model_checkpoint]

        start_time = time.time()
        # train the model
        if Ximg_val is not None:
            self.hist = self.model.fit(Ximg_train, yimg_train,
                                       validation_data=(Ximg_val, yimg_val),
                                       verbose=self.verbose,
                                       epochs=epochs,
                                       batch_size=mini_batch_size,
                                       callbacks=self.callbacks)
        else:
            self.hist = self.model.fit(Ximg_train, yimg_train,
                                       verbose=self.verbose,
                                       epochs=epochs,
                                       batch_size=mini_batch_size,
                                       callbacks=self.callbacks)

        self.duration = time.time() - start_time
        keras.models.save_model(self.model, self.output_directory + 'model.h5')
        if self.verbose:
            print('[D_ResNet] Training done!, took {}s'.format(self.duration))
        return

    def predict(self, X_img, y_img, best):
        if self.verbose:
            print('[D_ResNet] Predicting')
        if best:
            model = keras.models.load_model(self.output_directory + 'best_model.h5')
        else:
            model = keras.models.load_model(self.output_directory + 'model.h5')

        model_metrics, conf_mat, y_true, y_pred = predict_model(model, X_img, y_img, self.output_directory)
        save_logs(self.output_directory, self.hist, y_pred, y_true, self.duration)
        keras.backend.clear_session()

        if self.verbose:
            print('[D_ResNet] Prediction done!')

        return model_metrics, conf_mat
