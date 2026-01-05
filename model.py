"""CNN-LSTM model for Speech Emotion Recognition."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten,
    LSTM, BatchNormalization, Input, Reshape,
    GlobalAveragePooling2D, Bidirectional, TimeDistributed
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

from config import MODEL_PATH, BATCH_SIZE, EPOCHS, LEARNING_RATE, TEST_SIZE, RANDOM_STATE


def build_cnn_model(input_shape, num_classes):
    """Build a CNN model for emotion recognition."""
    model = Sequential([
        # First Conv Block
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second Conv Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third Conv Block
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Fourth Conv Block
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        
        # Dense layers
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model


def build_cnn_lstm_model(input_shape, num_classes):
    """Build a CNN-LSTM hybrid model for emotion recognition."""
    inputs = Input(shape=input_shape)
    
    # Reshape for TimeDistributed CNN
    # Split spectrogram into time segments
    x = Reshape((input_shape[0], input_shape[1], 1))(inputs)
    
    # CNN layers
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Reshape for LSTM: (batch, time_steps, features)
    shape = x.shape
    x = Reshape((shape[1], shape[2] * shape[3]))(x)
    
    # LSTM layers
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.3)(x)
    
    # Dense layers
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


def build_simple_lstm_model(input_shape, num_classes):
    """Build a simple LSTM model for 1D features."""
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model


class EmotionRecognizer:
    """Speech Emotion Recognition model wrapper."""
    
    def __init__(self, model_type='cnn'):
        self.model_type = model_type
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.num_classes = None
        self.input_shape = None
        
    def prepare_data(self, X, y):
        """Prepare data for training."""
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded)
        self.num_classes = len(self.label_encoder.classes_)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
        )
        
        # Normalize
        if self.model_type == 'cnn':
            # For CNN, normalize per sample
            X_train = (X_train - X_train.mean()) / (X_train.std() + 1e-8)
            X_test = (X_test - X_test.mean()) / (X_test.std() + 1e-8)
            # Add channel dimension
            X_train = X_train[..., np.newaxis]
            X_test = X_test[..., np.newaxis]
            self.input_shape = X_train.shape[1:]
        else:
            # For LSTM, use standard scaler
            orig_shape = X_train.shape
            X_train = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(orig_shape)
            X_test = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(orig_shape)
            self.input_shape = X_train.shape[1:]
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self):
        """Build the model based on type."""
        if self.model_type == 'cnn':
            self.model = build_cnn_model(self.input_shape, self.num_classes)
        elif self.model_type == 'cnn_lstm':
            self.model = build_cnn_lstm_model(self.input_shape[:-1], self.num_classes)
        else:
            self.model = build_simple_lstm_model(self.input_shape, self.num_classes)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the model."""
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(
                os.path.join(MODEL_PATH, f'best_model_{self.model_type}.keras'),
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model."""
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return loss, accuracy
    
    def predict(self, X):
        """Predict emotion from features."""
        if self.model_type == 'cnn':
            X = (X - X.mean()) / (X.std() + 1e-8)
            if X.ndim == 2:
                X = X[np.newaxis, ..., np.newaxis]
            elif X.ndim == 3:
                X = X[..., np.newaxis]
        else:
            X = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            if X.ndim == 1:
                X = X[np.newaxis, ...]
        
        predictions = self.model.predict(X, verbose=0)
        predicted_indices = np.argmax(predictions, axis=1)
        predicted_labels = self.label_encoder.inverse_transform(predicted_indices)
        confidences = np.max(predictions, axis=1)
        
        return predicted_labels, confidences, predictions
    
    def save(self, name='emotion_model'):
        """Save model and preprocessors."""
        self.model.save(os.path.join(MODEL_PATH, f'{name}.keras'))
        joblib.dump(self.label_encoder, os.path.join(MODEL_PATH, f'{name}_label_encoder.pkl'))
        joblib.dump(self.scaler, os.path.join(MODEL_PATH, f'{name}_scaler.pkl'))
        joblib.dump({
            'model_type': self.model_type,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes
        }, os.path.join(MODEL_PATH, f'{name}_config.pkl'))
        print(f"Model saved to {MODEL_PATH}")
    
    def load(self, name='emotion_model'):
        """Load model and preprocessors."""
        self.model = tf.keras.models.load_model(os.path.join(MODEL_PATH, f'{name}.keras'))
        self.label_encoder = joblib.load(os.path.join(MODEL_PATH, f'{name}_label_encoder.pkl'))
        self.scaler = joblib.load(os.path.join(MODEL_PATH, f'{name}_scaler.pkl'))
        config = joblib.load(os.path.join(MODEL_PATH, f'{name}_config.pkl'))
        self.model_type = config['model_type']
        self.input_shape = config['input_shape']
        self.num_classes = config['num_classes']
        print(f"Model loaded from {MODEL_PATH}")
