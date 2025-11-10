from typing import *
from sklearn.model_selection import train_test_split
import random as rd
import os
import pandas as pd
import shutil
from tqdm.notebook import tqdm
from PIL import Image, ImageEnhance, ImageOps
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pynvml import *
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json

json_file = "Credentials.json"
with open(json_file, "r") as f:
    credentials = json.load(f)

def get_vram_usage():
    """Gets the VRAM usage of the GPU in MB."""
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        used_vram_mb = info.used // (1024 ** 2)
        total_vram_mb = info.total // (1024 ** 2)
        nvmlShutdown()
        return used_vram_mb, total_vram_mb
    except NVMLError as error:
        print(error)
        return None, None
    finally:
        try:
            nvmlShutdown()
        except:
            pass

# For sending email notifications in case of errors on training
def send_email(text: str = ""):
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login(credentials["email"], credentials["password"])

    message = MIMEMultipart("alternative")
    message["Subject"] = "TRAINING CNN ERROR ALARM"
    message["From"] = ""
    message["To"] = ""
    text = f"""\
    Hi, this is an automated email from your syntax :3.
    The training has encountered an error.
    {text}
    """

    part1 = MIMEText(text, "plain")
    message.attach(part1)
    s.sendmail("", "", message.as_string())
    s.close()

def count_class(src, returnType: Literal["count", "full"]="count") -> pd.DataFrame:
    """
    Counts the number of instances in each class from the given source directory. BUT IT MUST BE ON `../DATA/`
    The source directory should contain subdirectories for each class, and each subdirectory should contain the images belonging to that class.

    Parameters
    ------------
    - src: 
        The source directory containing the class subdirectories.
    - returnType:
        The type of return value. If "count", it returns a list of counts for each class. 
        If "full", it returns a DataFrame with class names and their respective counts.
    """
    src_dir = os.path.join("../DATA", src)
    class_counts = []
    class_names = []

    # Get the list of class directories
    for class_name in os.listdir(src_dir):
        class_path = os.path.join(src_dir, class_name)
        if os.path.isdir(class_path):
            count = len(os.listdir(class_path))
            class_counts.append(count)
            class_names.append(class_name)
    
    if returnType == "count":
        return class_counts
    elif returnType == "full":
        df = pd.DataFrame({'Class': class_names, 'len': class_counts})
        df['len'] = df['len'].astype(int)
        return df

def split_train_eval(dst_name, composition = [0.85, 0.15], seed=rd.randint(0, 100000)):
    """
    Splits the dataset located in `../DATA/!FINAL/train_val` into train and val sets based on the specified composition.
    The train set will be created in `../DATA/!TEMP/<dst_name>/train` and the val set in `../DATA/!TEMP/<dst_name>/val`.

    Parameters
    ------------
    - dst_name:
        The name of the destination directory where the train and val sets will be created. 
        The structure will be `../DATA/!TEMP/<dst_name>/train` and `../DATA/!TEMP/<dst_name>/eval`.
    - composition:
        A list of two floats representing the proportions of the dataset to allocate to train and eval sets. 
        The values should sum to 1 (e.g., [0.9, 0.1]).
    - seed:
        Random seed for reproducibility.
    """
    temp = {}
    assert sum(composition) == 1, "Composition must sum to 1"
    
    # Split dataset into train and val equally on each class
    for class_name in tqdm(os.listdir(f'../DATA/!FINAL/train_val'), desc="Splitting data into train and eval"):
        class_path = os.path.join(f'../DATA/!FINAL/train_val', class_name)

        # Create paths for train and val directories
        for split in ["train", "val"]:
            os.makedirs(os.path.join("../DATA/!TEMP/", dst_name, split, class_name), exist_ok=True)
        # Split the images into train and eval sets
        train_image, eval_image = train_test_split(os.listdir(class_path), test_size=composition[1], random_state=seed)
        
        # Copy images to the respective directories
        for image in train_image:
            shutil.copy(os.path.join(class_path, image), os.path.join("../DATA/!TEMP", dst_name, "train", class_name, image))
        for image in eval_image:
            shutil.copy(os.path.join(class_path, image), os.path.join("../DATA/!TEMP", dst_name, "val", class_name, image))
        
        # Create a DataFrame with the number of images in each class for train and eval sets
        temp[class_name] = {
            "train": len(train_image),
            "val": len(eval_image)
        }
    df = pd.DataFrame(temp).T
    df.index.name = "Class"
    df.reset_index(inplace=True)
    df['train'] = df['train'].astype(int)
    df['val'] = df['val'].astype(int)
    return df

def adjustClassBalance(dir_name, target_sample_class = 100) -> pd.DataFrame:
    '''
    Adjusts the class balance issue in the dataset located in `DATA/!TEMP/<dir>/train`.

    If classes have fewer samples than target_sample_class, they will be oversampled by augmentation.
    
    If classes have more samples than target_sample_class, they will be undersampled by random selection.

    Parameters
    ------------
    - dir: 
        The directory name where the dataset is located, which should be in `DATA/!TEMP/<dir>`. Assumes the structure is `DATA/!TEMP/<dir>/train`.
    - target_sample_class: 
        The target number of samples per class.
    '''
    dir_path = os.path.join("../DATA/!TEMP", dir_name, "train")
    class_names = os.listdir(dir_path)
    old_len_classes = count_class(dir_path, returnType="count")

    for class_name in tqdm(class_names , desc="Adjusting class balance"):
        class_path = os.path.join(dir_path, class_name)
        images = os.listdir(class_path)
        current_len = len(images)

        # Oversampling if the class has fewer samples than target_sample_class
        if current_len < target_sample_class:
            index = 0
            while len(os.listdir(class_path)) < target_sample_class:
                image = rd.choice(images)
                image = Image.open(os.path.join(class_path, image))
                # random rotation
                rotation_angle = rd.randint(-20, 20) 
                image = image.rotate(rotation_angle)
                # Random horizontal flip
                if rd.random() > 0.5:
                    image = ImageOps.mirror(image)
                # zooming in or out
                if rd.random() > 0.5:
                    width, height = image.size
                    scale_factor = rd.uniform(0.8, 1.2)
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    image = image.resize((new_width, new_height))
                    image = image.crop((0, 0, width, height)) 
                # random brightness adjustment
                enhancer = ImageEnhance.Brightness(image)
                factor = rd.uniform(0.7, 1.3)
                image = enhancer.enhance(factor)

                # Save the augmented image
                saveName = f"augmented_{index}.jpg"
                image.save(os.path.join(class_path, saveName))
                index += 1
        
        # Undersampling if the class has more samples than target_sample_class 
        elif current_len > target_sample_class:
            selected_images = rd.sample(images, target_sample_class)
            for image in images:
                if image not in selected_images:
                    os.remove(os.path.join(class_path, image))
    
    new_len_classes = count_class(dir_path, returnType="count")

    return pd.DataFrame({
        "Class": class_names,
        "Old train Len": old_len_classes,
        "New train Len": new_len_classes
    })



def checkMetrics(model, imageGenerator, plot_confusion_matrix: Literal[True, False] = True, mode : Literal["simple", "detailed"] = "simple"):
    """
    Evaluates the model's performance on the given image generator and returns various metrics.
    The function calculates True Positives, False Positives, False Negatives, Precision, Recall, and F1 Score for each class.
    It also plots a confusion matrix if `plot_confusion_matrix` is set to True.
    The `mode` parameter determines the level of detail in the output:
    - "simple": Returns overall metrics without class-specific details.
    - "detailed": Returns overall metrics and class-specific metrics in a DataFrame format.
    The function assumes that the `imageGenerator` has been created using `ImageDataGenerator` and contains the class indices.

    Parameters
    ------------
    - model: 
        The trained Keras model to evaluate.
    - imageGenerator:
        The `ImageDataGenerator` instance that provides the images and their corresponding labels.
    - plot_confusion_matrix:
        A boolean indicating whether to plot the confusion matrix. Default is True.
    - mode:
        A string indicating the mode of operation. Can be "simple" or "detailed". Default is "simple".

    Returns
    ------------
    - If `mode` is "simple", returns a dictionary with overall metrics: 
        - "Total Instances": Total number of images in the dataset.
        - "True Positives": Total true positives across all classes.
        - "False Positives": Total false positives across all classes.
        - "False Negatives": Total false negatives across all classes.
        - "Precision": Overall precision of the model.
        - "Recall": Overall recall of the model.
        - "F1 Score": Overall F1 score of the model.
    """
    # Get true labels
    y_true = imageGenerator.classes
    # Get predicted class
    y_pred_prob = model.predict(imageGenerator, verbose=1)
    y_pred = y_pred_prob.argmax(axis=-1)

    # Get class mappings
    class_labels = imageGenerator.class_indices
    class_labels = {v: k for k, v in class_labels.items()}
    total_image = len(y_true)
    returnClassmetrics = {}

    # Calculate metrics for each class
    for class_index, class_name in class_labels.items():
        # Get true positives, false positives, and false negatives
        tp = sum((y_true == class_index) & (y_pred == class_index))
        fp = sum((y_true != class_index) & (y_pred == class_index))
        fn = sum((y_true == class_index) & (y_pred != class_index))
        # Calculate precision, recall, and F1 score
        class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0

        returnClassmetrics[class_name] = {
            'True Positives': int(tp),
            "False Positives": int(fp),
            "False Negatives": int(fn),
            'Precision': round(class_precision,4),
            'Recall': round(class_recall, 4),
            'F1 Score': round(class_f1,4)
        }
    # Calculate overall metrics
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    overal_precision = np.mean([metrics['Precision'] for metrics in returnClassmetrics.values()])
    overall_recall = np.mean([metrics['Recall'] for metrics in returnClassmetrics.values()])
    overall_f1_score = np.mean([metrics['F1 Score'] for metrics in returnClassmetrics.values()])

    if plot_confusion_matrix:
        cm = confusion_matrix(y_true, y_pred)
        # Plot confusion matrix
        classes = list(class_labels.values())
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()
        
    if mode == "detailed":
        result = { 
            "Balanced Accuracy": round(balanced_accuracy, 4),
            "Precision": round(overal_precision, 4), 
            "Recall": round(overall_recall, 4), 
            "F1 Score": round(overall_f1_score, 4), 
            "Class Metrics": returnClassmetrics
        } 
        # Create a main DataFrame with overall metrics
        overall_metrics = {k: [v] for k, v in result.items() if k != "Class Metrics"}
        overall_df = pd.DataFrame(overall_metrics)
        print("Overall Metrics:")
        display(overall_df)

        # Create a DataFrame for class-specific metrics
        class_metrics = result["Class Metrics"]
        class_df = pd.DataFrame.from_dict(class_metrics, orient='index').sort_index()
        print("\nClass-Specific Metrics:")
        display(class_df)
        
    if mode == "simple":
        result = { 
            "Balanced Accuracy": round(balanced_accuracy, 4),
            "Precision": round(overal_precision, 4), 
            "Recall": round(overall_recall, 4), 
            "F1 Score": round(overall_f1_score, 4), 
        } 
    return result

# def model_history_plot(history):
#   '''
#   Plot the training and validation loss and accuracy over the epochs.
#   Params:
#     history: keras.callbacks.History: history object of the model
#   '''
#   fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 8))
#   axes[0].plot(history.history['loss'], label='Training Loss', marker='o')
#   axes[0].plot(history.history['val_loss'], label='Validation Loss', marker='o')
#   axes[0].set_title('Loss over Epochs', fontsize=14, fontweight='bold')
#   axes[0].set_xlabel('Epoch', fontsize=12)
#   axes[0].set_ylabel('Loss', fontsize=12)
#   axes[0].grid(True)
#   axes[0].legend()

#   axes[1].plot(history.history['accuracy'], label='Training Accuracy', marker='o')
#   axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
#   axes[1].set_title('Accuracy over Epochs', fontsize=14, fontweight='bold')
#   axes[1].set_xlabel('Epoch', fontsize=12)
#   axes[1].set_ylabel('Accuracy', fontsize=12)
#   axes[1].grid(True)
#   axes[1].legend()

#   plt.tight_layout()
#   plt.show()

def get_test_generator(batch_size=32, target_size=(224, 224)):
    """
    Creates a Keras ImageDataGenerator instance for the test dataset.
    
    Parameters
    ------------
    - batch_size: 
        The size of the batches of data (default is 32).
    - target_size: 
        The size to which all images found will be resized (default is (224, 224)).
    
    Returns
    ------------
    - A Keras ImageDataGenerator instance for the test dataset.
    """
    DIR = os.path.join("../DATA/!FINAL/test")

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        DIR,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return test_generator

def get_train_eval_Generator(DIR, batch_size=32, target_size=(224, 224)):
    """
    Creates Keras ImageDataGenerator instances for training and validation datasets.
    
    Parameters
    ------------
    - DIR: 
        The directory containing the images. Assumes the structure is `DATA/!TEMP/<DIR>/train` and `DATA/!TEMP/<DIR>/val`.
    - batch_size: 
        The size of the batches of data (default is 32).
    - target_size: 
        The size to which all images found will be resized (default is (224, 224)).
    
    Returns
    ------------
    - A Keras ImageDataGenerator instance.
    """
    DIR = os.path.join("../DATA/!TEMP", DIR)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        # No augmentation for training data. Augmentation has been done in `adjustClassBalance`.
    )
    val_datagen = ImageDataGenerator(rescale=1./255)


    train_generator = train_datagen.flow_from_directory(
        os.path.join(DIR, "train"),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    validation_generator = val_datagen.flow_from_directory(
        os.path.join(DIR, "val"),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, validation_generator

def get_model(num_classes, 
              conv2d_filters: List[int] = [32, 64, 128, 256, 512], 
              num_conv_blocks: int = 5, 
              dropout_rate: float = 0.5, 
              optimizer: Literal["adam", "rmsprop"] = "adam", 
              learning_rate: float = 0.0001) -> keras.Sequential:
    """
    Creates a somewhat simple multi-layer CNN model using Keras.
    The model consists of multiple convolutional blocks, each followed by a max pooling layer.
    The number of convolutional blocks and the number of filters in each block can be specified.

    Parameters
    ------------
    - num_classes: 
        The number of output classes for the classification task.
    - conv2d_filters:
        A list of integers specifying the number of filters for each convolutional layer.
        The length of this list should be at least equal to `num_conv_blocks`.
    - num_conv_blocks:
        The number of convolutional blocks to be added to the model. Each block consists of a Conv2D layer followed by a MaxPooling2D layer.
    - dropout_rate:
        The dropout rate to be used in the dropout layer.
    - optimizer:
        The optimizer to be used for compiling the model. Can be "adam" or "rmsprop".
    - learning_rate:
        The learning rate for the optimizer. Default is 0.0001.
    
    Returns
    ------------
    - A Keras Sequential model with the specified architecture.
    """

    # Validate input parameters
    if len(conv2d_filters) < num_conv_blocks:
        raise ValueError(f"conv2d_filters must have at least {num_conv_blocks} elements, got {len(conv2d_filters)}")
    if num_conv_blocks < 1:
        raise ValueError("num_conv_blocks must be at least 1")
    
    # The block! No batch_normalization or those fancy stuffs!
    def conv_block(filters, kernel_size=(3, 3), activation='relu', padding='same'):
        return keras.Sequential([
            keras.layers.Conv2D(filters, kernel_size, activation=activation, padding=padding),
            keras.layers.MaxPooling2D((2, 2))
        ])
  
    model = keras.Sequential()
    # Input layer
    model.add(keras.layers.InputLayer(input_shape=(224, 224, 3)))
    # Add convolutional blocks with specified filters
    for i in range(num_conv_blocks):
        model.add(conv_block(conv2d_filters[i]))

    # Flatten the output and add dense layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    if optimizer == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        # optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=["accuracy", keras.metrics.F1Score( average='macro', name='f1score')]
    )

    return model