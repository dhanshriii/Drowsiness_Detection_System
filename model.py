import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Configuration parameters - OPTIMIZED
IMG_SIZE = 96  # Reduced from 145 to 96
BATCH_SIZE = 64  # Increased from 32 to 64
EPOCHS = 10  # Reduced from 15 to 10
DATA_DIR = r"/content/drive/MyDrive/Data"

def create_efficient_model():
    """Create a more efficient model using transfer learning with MobileNetV2"""
    # Use MobileNetV2 as base model - very efficient architecture
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create the model
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)  # Smaller dense layer
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    # Compile with a slightly higher learning rate for faster convergence
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data():
    # Data augmentation for training set - using simpler augmentations
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,  # Reduced from 10
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    # Only rescaling for validation set
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Setting up training generators with optimized caching and prefetching
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['Active Subjects', 'Fatigue Subjects'],
        subset='training'
    )
    
    # Setting up validation generators
    validation_generator = val_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['Active Subjects', 'Fatigue Subjects'],
        subset='validation'
    )
    
    return train_generator, validation_generator

def train_model():
    # Enable mixed precision training for faster computation on T4 GPU
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Ensure output directory exists
    os.makedirs('models', exist_ok=True)
    
    # Prepare data generators
    train_generator, validation_generator = prepare_data()
    
    # Create the optimized model
    model = create_efficient_model()
    
    # Display model summary
    model.summary()
    
    # Setup callbacks - more aggressive early stopping
    checkpoint = ModelCheckpoint(
        'models/drowsiness_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,  # Reduced from 5 to 3
        restore_best_weights=True,
        verbose=1
    )
    
    # Calculate steps per epoch to avoid partial batches
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = validation_generator.samples // BATCH_SIZE
    
    # Train the model with efficient settings
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping]
       
    )
    
    # Save the final model
    model.save('models/drowsiness_detection_final.h5')
    
    # Plot training results
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.show()
    
    print("Model training completed and saved to 'models/drowsiness_detection_final.h5'")
    
    # Print approximate training time
    print(f"Training completed in {len(history.history['loss'])} epochs")
    
    return model, history

def fine_tune_model(model, train_generator, validation_generator):
    """Optional: Fine-tune the pre-trained model after initial training"""
    # Only run this after the initial training shows good results
    # Unfreeze some of the top layers of the base model
    base_model = model.layers[1]
    base_model.trainable = True
    
    # Freeze all layers except the top 10
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Calculate steps
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = validation_generator.samples // BATCH_SIZE
    
    # Train for a few more epochs
    history_fine = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=5,
        workers=4,
        use_multiprocessing=True,
        max_queue_size=10
    )
    
    # Save the fine-tuned model
    model.save('models/drowsiness_detection_finetuned.h5')
    
    return model, history_fine

if __name__ == "__main__":
    # Train the initial model
    model, history = train_model()
    
    # If you have time and want better accuracy, uncomment to run fine-tuning
    # train_generator, validation_generator = prepare_data()
    # model, history_fine = fine_tune_model(model, train_generator, validation_generator)