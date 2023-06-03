import time
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Conv1D, BatchNormalization, ELU, Dense, Reshape, concatenate
from classifiers.classifiers import predict_model
from utils.classifier_tools import create_class_weight
from utils.tools import save_logs



class Classifier_MC_CNN:
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False):
        if verbose:
            print('Creating MC_CNN Classifier')
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
        X = Conv2D(8, (5, 1), strides=1, padding="valid", activation='sigmoid')(X_input)
        X = keras.layers.AveragePooling2D((2, 1), padding='valid')(X)
        X = Conv2D(5, (5, 1), strides=1, padding="valid", activation='sigmoid')(X)
        X = keras.layers.AveragePooling2D((2,1), padding='valid')(X)
        flatten = keras.layers.Flatten()(X)

        dense = Dense(128, activation="relu")(flatten)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(dense)
        model = keras.models.Model(inputs=X_input, outputs=output_layer)
        return model

    def fit(self, Ximg_train, yimg_train, Ximg_val, yimg_val, epochs, batch_size):
        if self.verbose:
            print('[MC_CNN] Training MC_CNN Classifier')

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
        print('[MC_CNN] Training done!, took {}s'.format(self.duration))

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
