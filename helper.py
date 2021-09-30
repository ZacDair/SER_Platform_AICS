import os


# # Search dirs and subdirs
# for root, dirs, files in os.walk("F:\emo_detect\datasources\SAVEE\SAVEE_DATA"):
#     for file in files:
#         newName = os.path.join(root, file).split("\\")[6]
#         if file[0] in ['a', 'd', 'f', 'h', 'n']:
#             newName = newName + file[0] + file
#         else:
#             newName = newName + file
#
#         os.rename(os.path.join(root, file), os.path.join(root, newName))
#
#
# # Create a CNN model using the specified structure
# def model_create_CNN(inputShape, outputShape):
#     model = Sequential()
#     model.add(Conv1D(128, 5, padding='same',
#                      input_shape=inputShape))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.2))
#     # model.add(MaxPooling1D(pool_size=(8)))
#     model.add(Conv1D(128, 5, padding='same'))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.2))
#     model.add(Flatten())
#     model.add(Dense(outputShape))
#     model.add(Activation('softmax'))
#
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='rmsprop',
#                   metrics=['accuracy'])
#     return model


