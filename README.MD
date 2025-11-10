# Image classification of oil palm caterpillars using machine learning and deep learning approaches

This repository contains code and resources for classifying oil palm caterpillars using SVM, HOG-SVM, and CNN models which I made my Thesis on. The datasets were collected first-hand from oil palm plantations and scraped the internet. I ain't including the datasets as many were taken from copyrighted sources.

## Abstract


Oil palm is an important commodity in Indonesia, with one of the main challenges being caterpillar pest attacks. Identifying pest species is a crucial step in preventing such attacks. Therefore, an effective method for species classification is required. This study applies image classification using three approaches: machine learning with Support Vector Machine (SVM), Histogram of Oriented Gradients as a feature extraction technique combined with SVM (HOG-SVM), and deep learning with Convolutional Neural Network (CNN). Each approach was optimized through hyperparameter tuning using Optuna and validated using stratified k-fold cross-validation. Model performance was evaluated using the macro average F1-score and prediction time for a single image. The results show that CNN achieved the best performance with an F1-score of 90% and prediction time of 0.0105 seconds. HOG-SVM obtained an F1-score of 69% with a prediction time of 0.0006 seconds, while SVM only reached 52% with a prediction time of 0.0403 seconds. These findings indicate that CNN excels in handling image data, whereas HOG-SVM can serve as an efficient alternative under limited computational resources.

Keywords: CNN, HOG-SVM, hyperparameter tuning, image classification, SVM.


## Repository Structure
The repository is organized as follows:
```
├── DATA/                   # Directory for datasets (not included due to copyright)
├── UTILS/                  # Utility functions for data scraping and preprocessing
├── SRC/                    # Source code for model training and evaluation, statistical analysis, visualization, etc.
├── PANDUAN.pdf             # Guide on classifying oil palm caterpillars manually
└── README.md               # This README file
```

## Note
Just use the code for learning purposes. The datasets are not included in this repository due to copyright restrictions. If you wish to replicate the study, please collect your own datasets from oil palm plantations or other sources. Thank you for understanding xoxo!