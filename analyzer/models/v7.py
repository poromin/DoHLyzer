from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU, Flatten

def create_model(segment_size):
    model = Sequential()
    model.add(GRU(segment_size * 8, input_shape=(segment_size, 5), activation='relu', return_sequences=True))
    model.add(GRU(segment_size * 4, activation='relu'))
    model.add(Dense(segment_size * 6, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(segment_size * 2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
