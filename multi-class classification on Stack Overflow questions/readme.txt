```markdown
Text Classification with TensorFlow


This project demonstrates a text classification pipeline using TensorFlow, including data loading, preprocessing, model training, evaluation, and exporting. The pipeline includes two types of text vectorization approaches: binary and integer encoding. The code covers the following key steps:


1. Loading and preparing the dataset.
2. Defining and training models with binary and integer vectorized inputs.
3. Evaluating the models.
4. Exporting the trained model with integrated text preprocessing.


Project Overview


The project uses the Stack Overflow dataset, which is processed to classify text data into predefined categories. The code includes a basic binary classification model and a more sophisticated integer vectorized model using convolutional layers.


Setup


Prerequisites


- Python 3.x
- TensorFlow 2.x
- Matplotlib


You can install the necessary Python packages using pip:


```bash
pip install tensorflow matplotlib
```


Dataset


The dataset used in this project is obtained from Stack Overflow and is expected to be in a directory structure suitable for `tf.keras.utils.text_dataset_from_directory`. Make sure the dataset is placed in the `C:\\Users\\admin\\Music\\gaurav` directory.


Usage


 Load and Prepare Data


The dataset is loaded from a directory and split into training, validation, and test datasets. The `TextVectorization` layers are used to preprocess text data.


 Define and Train Models


Two models are defined:


1. **Binary Model**: A simple dense network that uses binary vectorized input.
2. **Integer Model**: A convolutional neural network that uses integer vectorized input.


Both models are trained on the training dataset and validated on the validation dataset.


Evaluate Models


After training, the models are evaluated on the test dataset. The performance metrics include accuracy and loss.


Export Model


An exportable model is created, which includes text preprocessing and the classification layers. This model can be used for inference on new data.


Code Walkthrough


 1. Loading the Dataset


The dataset is loaded using `tf.keras.utils.text_dataset_from_directory`, which supports text classification tasks.


```python
raw_train_ds = tf.keras.utils.text_dataset_from_directory(train_dir, ...)
raw_val_ds = tf.keras.utils.text_dataset_from_directory(train_dir, ...)
raw_test_ds = tf.keras.utils.text_dataset_from_directory('test', ...)
```


 2. Text Vectorization


Two types of vectorization are applied:


- **Binary Vectorization**: Converts text into binary format.
- **Integer Vectorization**: Converts text into sequences of integers.


```python
binary_vectorize_layer = layers.TextVectorization(max_tokens=10000, output_mode='binary')
int_vectorize_layer = layers.TextVectorization(max_tokens=10000, output_mode='int', output_sequence_length=250)
```


 3. Model Definitions


The binary model and integer model are defined and compiled.


```python
binary_model = tf.keras.Sequential([layers.Dense(4)])
int_model = create_model(vocab_size=max_size_vocab + 1, num_labels=4)
```


4. Training and Evaluation


The models are trained and evaluated with accuracy and loss plots.


```python
history = binary_model.fit(binary_train_ds, validation_data=binary_val_ds, epochs=10)
history = int_model.fit(int_train_ds, validation_data=int_val_ds, epochs=5)
```
 5. Export Model


The model is exported with integrated text preprocessing.


```python
export_model = tf.keras.Sequential([
    vectorize_layer,
    tf.keras.Sequential([
        layers.Embedding(max_size_vocab + 1, 64, mask_zero=True),
        layers.Conv1D(64, 5, padding="valid", activation="relu", strides=2),
        layers.GlobalMaxPooling1D(),
        layers.Dense(4, activation='softmax')
    ])
])
```




```