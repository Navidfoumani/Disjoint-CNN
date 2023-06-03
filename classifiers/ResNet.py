import time

from tensorflow import keras

from classifiers.classifiers import predict_model
from utils.tools import save_logs

# Most of the code here are taken from https://github.com/hfawaz/dl-4-tsc

class Classifier_ResNet:
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False):
        if verbose:
            print('[ResNet] Creating ResNet Classifier')
        self.verbose = verbose
        self.output_directory = output_directory
        self.model = self.build_model(input_shape, nb_classes)
        if verbose:
            self.model.summary()

        self.model.save_weights(self.output_directory + 'model_init.h5')

    def build_model(self, input_shape, nb_classes):
        n_feature_maps = 64
        input_layer = keras.layers.Input(input_shape)

        # BLOCK 1
        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        # FINAL

        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        return model

    def fit(self, Ximg_train, yimg_train, Ximg_val=None, yimg_val=None, epochs=10, batch_size=16):
        if self.verbose:
            print('[ResNet] Training ResNet Classifier')
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
            print('[ResNet] Training done!, took {}s'.format(self.duration))
        return

    def predict(self, X_img, y_img, best):
        if self.verbose:
            print('[ResNet] Predicting')
        if best:
            model = keras.models.load_model(self.output_directory + 'best_model.h5')
        else:
            model = keras.models.load_model(self.output_directory + 'model.h5')

        model_metrics, conf_mat, y_true, y_pred = predict_model(model, X_img, y_img,
                                                                              self.output_directory)
        save_logs(self.output_directory, self.hist, y_pred, y_true, self.duration)

        keras.backend.clear_session()

        if self.verbose:
            print('[ResNet] Prediction done!')

        return model_metrics, conf_mat
