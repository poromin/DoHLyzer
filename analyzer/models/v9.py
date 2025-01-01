from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

def create_model(segment_size):
    model = Sequential()
    model.add(Dense(segment_size * 10, input_shape=(segment_size, 5), activation='relu'))
    model.add(Flatten())
    model.add(Dense(segment_size * 6, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(segment_size * 2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model