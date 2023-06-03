import time
import keras
import numpy as np

from keras.layers import Input, Conv2D, Conv1D, BatchNormalization, ELU, Dense, Reshape, concatenate
from keras.layers import MaxPooling2D, MaxPooling1D, GlobalAveragePooling2D, GlobalAveragePooling1D
from classifiers.classifiers import predict_model
from utils.classifier_tools import create_class_weight
from utils.tools import save_logs



class Classifier_T_CNN:
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False):
        if verbose:
            print('Creating T_CNN Classifier')
        self.verbose = verbose
        self.output_directory = output_directory
        # Build Model -----------------------------------------------------------
        self.model = self.build_model(input_shape, nb_classes)
        # -----------------------------------------------------------------------
        if verbose:
            self.model.summary()
        self.model.save_weights(self.output_directory + 'model_init.h5')

    def build_model(self, input_shape, nb_classes):

        X_input = Input(shape=input_shape)
        # Temporal Convolutions
        X = Conv2D(64, (8, 1), strides=1, padding="same", kernel_initializer='he_uniform')(X_input)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.ELU(alpha=1.0)(X)

        MaxPool = MaxPooling2D(pool_size=(5, 1), strides=None, padding='valid')(X)
        gap_DCNN = GlobalAveragePooling2D()(MaxPool)

        dense = Dense(128, activation="relu")(gap_DCNN)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(dense)
        model = keras.models.Model(inputs=X_input, outputs=output_layer)
        return model

    def fit(self, Ximg_train, yimg_train, Ximg_val, yimg_val, epochs, batch_size):
        if self.verbose:
            print('[T_CNN] Training T_CNN Classifier')

        mini_batch_size = int(min(Ximg_train.shape[0] / 10, batch_size))
        self.model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.0001)
        file_path = self.output_directory + 'best_model.h5'
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)
        self.callbacks = [reduce_lr, model_checkpoint]

        # create class weights based on the y label proportions for each class
        class_weight = create_class_weight(yimg_train)

        start_time = time.time()

        self.hist = self.model.fit(Ximg_train, yimg_train,
                                   validation_data=[Ximg_val, yimg_val],
                                   class_weight=class_weight,
                                   verbose=0,
                                   epochs=epochs,
                                   batch_size=mini_batch_size,
                                   callbacks=self.callbacks)
        self.duration = time.time() - start_time
        '''
        self.hist = self.model.fit(Ximg_train, yimg_train,
                                   validation_split=0.1,
                                   class_weight=class_weight,
                                   verbose=self.verbose,
                                   epochs=epochs,
                                   batch_size=mini_batch_size,
                                   callbacks=self.callbacks)
        self.duration = time.time() - start_time
        '''
        keras.models.save_model(self.model, self.output_directory + 'model.h5')
        print('[T_CNN] Training done!, took {}s'.format(self.duration))

    def predict(self, X_img, y_img, best):
        if best:
            print(self.output_directory)
            model = keras.models.load_model(self.output_directory + 'best_model.h5')
        else:
            model = keras.models.load_model(self.output_directory + 'model.h5')

        model_metrics, conf_mat, y_true, y_pred = predict_model(model, X_img, y_img, self.output_directory)
        save_logs(self.output_directory, self.hist, y_pred, y_true, self.duration)
        keras.backend.clear_session()
        return model_metrics, conf_mat
