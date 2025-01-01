from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPool1D, Flatten

def create_model(segment_size):
    model = Sequential()
    model.add(Conv1D(filters=segment_size * 4, kernel_size=3, input_shape=(segment_size, 5), activation='relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(LSTM(segment_size * 8, activation='relu', return_sequences=True))
    model.add(Flatten())
    model.add(Dense(segment_size * 6, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(segment_size * 2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
