import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
import wandb
from wandb.keras import WandbCallback



#Class for ResNet
class RESNET:

  # Constructor
  def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True):
        self.output_directory = output_directory
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose

        return
  #Building model (cf https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py)
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

        model = keras.models.Model(inputs=input_layer, outputs=output_layer, name='ResNet')

        return model

  #Complie model, we use ReduceLR to stop the learning


  def lr_scheduler(epoch, lr):
    # log the current learning rate onto W&B
    if wandb.run is None:
        raise wandb.Error("You must call wandb.init() before WandbCallback()")


    wandb.log({'learning_rate': lr}, commit=False)

    return lr


  def compile(self):

        self.model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=[keras.metrics.Accuracy(),keras.metrics.Recall(), keras.metrics.Precision()])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.0001)





        self.callbacks = [tf.keras.callbacks.LearningRateScheduler(reduce_lr), tf.keras.callbacks.LearningRateScheduler(self.lr_scheduler)]
        print('=== Compiled ===')

        wandb.login(key="89972c25af0c49a4e2e1b8663778daedd960634a")
        wandb.init(project="ImbalanceClassification", entity="djbd")
        #wandb.run.name = f"MPPO_{dateenv}_{wandb.run.id}"

        print('=== Connected to wandb ===')

  # fit model
  def fit(self, x_train, y_train, x_val, y_val):
        #Test for GPU avaible, if not return error
        if not tf.test.is_gpu_available:
            print('error')
            exit()

        batch_size = 64
        nb_epochs = 1500


        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        start_time = time.time()

        wandb.config = {
            "learning_rate": 0.001,
            "epochs": nb_epochs,
            "batch_size": mini_batch_size
      }

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=[WandbCallback(), self.callbacks[0], self.callbacks[1]])

        duration = time.time() - start_time
        print(f'=== Fitted in {duration} secondes')

        self.model.save(self.output_directory + 'ResNet_weights.hdf5')



        return [hist, duration]

  # Prediction
  def predict(self, x_test):
      # Return array with each class prediction (not in One hot encoding)
      return np.argmax(self.model.predict(x_test), axis = 1)
