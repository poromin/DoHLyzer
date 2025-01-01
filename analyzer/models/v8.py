from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional

def create_model(segment_size):
    model = Sequential()
    model.add(Bidirectional(LSTM(segment_size * 8, activation='relu', return_sequences=True), input_shape=(segment_size, 5)))
    model.add(LSTM(segment_size * 4, activation='relu'))
    model.add(Dense(segment_size * 6, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(segment_size * 2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
