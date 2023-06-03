import time
import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ELU, Dense, Reshape, concatenate, Permute
from tensorflow.keras.layers import GlobalAveragePooling1D, Dropout, Activation, multiply
from tensorflow.keras.layers import LSTM
from classifiers.classifiers import predict_model
from utils.classifier_tools import create_class_weight
from utils.tools import save_logs
# from keras_multi_head import MultiHeadAttention


class Classifier_MLSTM_FCN:
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False):
        if verbose:
            print('Creating MLSTM_FCN Classifier')
        self.verbose = verbose
        self.output_directory = output_directory
        # Build Model -----------------------------------------------------------
        self.model = self.build_model(input_shape, nb_classes)
        # -----------------------------------------------------------------------
        if verbose:
            self.model.summary()
        self.model.save_weights(self.output_directory + 'model_init.h5')

    def build_model(self, input_shape, nb_classes):

        Y_input = Input(shape=input_shape)
        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(Y_input)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = squeeze_excite_block(y)

        y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = squeeze_excite_block(y)

        y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = GlobalAveragePooling1D()(y)

        x = Permute((2, 1))(Y_input)
        x = LSTM(8)(x)
        x = Dropout(0.8)(x)

        x = concatenate([x, y])

        out = Dense(nb_classes, activation='softmax')(x)
        model = tensorflow.keras.models.Model(inputs=Y_input, outputs=out)

        return model



    def fit(self, Ximg_train, yimg_train, Ximg_val, yimg_val, epochs, batch_size):

        mini_batch_size = int(min(Ximg_train.shape[0] / 10, batch_size))
        self.model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.0001)
        file_path = self.output_directory + 'best_model.h5'
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)
        self.callbacks = [reduce_lr, model_checkpoint]

        # create class weights based on the y label proportions for each image
        class_weight = create_class_weight(yimg_train)
        start_time = time.time()
        # train the model
        '''
        self.hist = self.model.fit(Ximg_train, yimg_train,
                                   validation_data=[Ximg_val, yimg_val],
                                   class_weight=class_weight,
                                   verbose=self.verbose,
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

        keras.models.save_model(self.model, self.output_directory + 'model.h5')
        print('[MLSTM_FCN] Training done!, took {}s'.format(self.duration))
        keras.backend.clear_session()

    def predict(self, X_img, y_img, best):
        if best:
            model = keras.models.load_model(self.output_directory + 'best_model.h5')
        else:
            model = keras.models.load_model(self.output_directory + 'model.h5')

        model_metrics, conf_mat, y_true, y_pred = predict_model(model, X_img, y_img, self.output_directory)
        save_logs(self.output_directory, self.hist, y_pred, y_true, self.duration)
        keras.backend.clear_session()
        return model_metrics, conf_mat

def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    filters = input.shape[-1]  # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se
