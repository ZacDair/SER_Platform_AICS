import os


# Rename SAVEE Files in a useable format
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
# Original Model Structure
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


# # Rename SUBESCO Files in a useable format - both required due to sentence num
# for root, dirs, files in os.walk("F:\emo_detect\datasources\SUBESCO\SUBESCO_DATA"):
#     for file in files:
#         filenameParts = file.split("_")
#         newName = str(filenameParts[0]) + "_" + str(filenameParts[1]) + "_" + str(filenameParts[3]) + str(filenameParts[4]) + "_" + str(filenameParts[5])[0:2] + "_" + str(filenameParts[6])
#         os.rename(os.path.join(root, file), os.path.join(root, newName))

# for root, dirs, files in os.walk("F:\emo_detect\datasources\SUBESCO\SUBESCO_DATA"):
#     for file in files:
#         filenameParts = file.split("_")
#         if len(str(filenameParts[2])) == 2:
#             sentenceNum = "S0" + str(filenameParts[2])[1]
#         else:
#             sentenceNum = str(filenameParts[2])
#
#         newName = str(filenameParts[0]) + "_" + str(filenameParts[1]) + "_" + sentenceNum + "_" + str(filenameParts[3]) + "_" + str(filenameParts[4])
#         print(newName)
#         os.rename(os.path.join(root, file), os.path.join(root, newName))