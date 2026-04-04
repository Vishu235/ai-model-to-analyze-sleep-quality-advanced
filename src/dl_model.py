from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten,
    TimeDistributed, LSTM, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam

def build_cnn_lstm(input_shape, num_classes):
    inp = Input(shape=input_shape)
    
    x = TimeDistributed(Conv1D(filters=64, kernel_size=64, strides=8, activation="relu", padding="same"))(inp)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling1D(pool_size=4))(x)
    x = TimeDistributed(Dropout(0.3))(x) # Added dropout inside the feature extractor

    x = TimeDistributed(Conv1D(filters=128, kernel_size=8, strides=1, activation="relu", padding="same"))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling1D(pool_size=4))(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(128, dropout=0.5, recurrent_dropout=0.0)(x) 

    x = Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)

    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inp, out)
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model