"""Training script for Speech Emotion Recognition model."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

from data_loader import load_ravdess_data, save_features, load_saved_features
from model import EmotionRecognizer
from config import MODEL_PATH, FEATURES_PATH, EMOTIONS


def plot_training_history(history, save_path=None):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def train_model(model_type='cnn', use_cached_features=True):
    """
    Train the emotion recognition model.
    
    Args:
        model_type: 'cnn', 'cnn_lstm', or 'lstm'
        use_cached_features: Whether to use cached features if available
    """
    print("=" * 60)
    print(f"Training {model_type.upper()} Model")
    print("=" * 60)
    
    # Load or extract features
    feature_prefix = 'ravdess_mel' if model_type in ['cnn', 'cnn_lstm'] else 'ravdess_combined'
    feature_file = os.path.join(FEATURES_PATH, f'{feature_prefix}_X.npy')
    
    if use_cached_features and os.path.exists(feature_file):
        print("Loading cached features...")
        X, y, metadata = load_saved_features(prefix=feature_prefix)
    else:
        print("Extracting features...")
        use_mel = model_type in ['cnn', 'cnn_lstm']
        X, y, metadata = load_ravdess_data(use_mel_spec=use_mel)
        save_features(X, y, metadata, prefix=feature_prefix)
    
    print(f"\nData shape: {X.shape}")
    print(f"Labels: {np.unique(y)}")
    
    # Initialize model
    recognizer = EmotionRecognizer(model_type=model_type)
    
    # Prepare data
    print("\nPreparing data...")
    X_train, X_test, y_train, y_test = recognizer.prepare_data(X, y)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Build model
    print("\nBuilding model...")
    model = recognizer.build_model()
    model.summary()
    
    # Train model
    print("\nTraining model...")
    history = recognizer.train(X_train, y_train, X_test, y_test)
    
    # Evaluate
    print("\nEvaluating model...")
    loss, accuracy = recognizer.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Get predictions for detailed metrics
    y_pred_labels, confidences, _ = recognizer.predict(X_test[..., 0] if model_type == 'cnn' else X_test)
    y_true_labels = recognizer.label_encoder.inverse_transform(np.argmax(y_test, axis=1))
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true_labels, y_pred_labels))
    
    # Plot results
    plot_training_history(history, save_path=os.path.join(MODEL_PATH, f'{model_type}_training_history.png'))
    plot_confusion_matrix(
        y_true_labels, y_pred_labels,
        labels=recognizer.label_encoder.classes_,
        save_path=os.path.join(MODEL_PATH, f'{model_type}_confusion_matrix.png')
    )
    
    # Save model
    recognizer.save(name=f'emotion_model_{model_type}')
    
    return recognizer, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Speech Emotion Recognition Model')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'cnn_lstm', 'lstm'],
                        help='Model type to train')
    parser.add_argument('--no-cache', action='store_true', help='Do not use cached features')
    
    args = parser.parse_args()
    
    recognizer, history = train_model(
        model_type=args.model,
        use_cached_features=not args.no_cache
    )
