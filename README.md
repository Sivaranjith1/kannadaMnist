# Kannada Mnist
A convolutional neural network for predicting one of the following kannada letters in a image. The model were 99% accurate on the test dataset from kaggle

![alt text][eksampel]

## Table of content
1.  [Usage](#usage)
    1.  [Required packages](#packages)
    1.  [Input data](#inputData)
1.  [Dataset](#dataset)
1.  [Training](#training)
1.  [Credits](#credits)

<a name="usage"></a>
## Usage
The trained weights are saved in Kannada-Mnist.model. To use this in your  project add Kannada-Mnist to your local repo.




[eksampel]: https://storage.googleapis.com/kaggle-media/competitions/Kannada-MNIST/kannada.png "Possible Kannada signs"
```python
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('Kannada-Mnist.model')

letters = ['omdu', 'eradu', 'muru', 'nalku', 'aidu', 'aru', 'elu', 'emtu', 'ombattu', 'hattu']
prediction = model.predict(np.asarray(X/255.0)) #this will return the index of the letter

print(letters[prediction[0]]) #will print out the letter
``` 

<a name="packages"></a>
### Required packages
To install the required packages, run the following command in the consol:
```script
pip install -r requirement.txt
```

<a name="inputData"></a>
### Input data
The input data of the model is expected to be a numpy array with shape of (x, 28,28,1). Which means that the input should be a array of 28x28 gray scale images of a singel letter.

<a name="dataset"></a>
## Dataset
The dataset is made by Vinay Uday Prabhu and uploaded to kaggle. https://www.kaggle.com/c/Kannada-MNIST/overview. To use the dataset for training download the dataset and save the csv file in a folder named dataset/

<a name="training"></a>
## Training
1.  Download the dataset from kaggle and save it in a file named dataset/
2.  Run dataPreprossessing.py to create a dataset file, named X.pickle, and a label file, named Y.pickle.
3.  To train the model, open up convnet.py and choose a name by changing the NAME variable
    ```python
    NAME = "Kannada-Mnist"
    ```
    1. Run the file named convnet.py
    2. The model will now be saved in a file that starts with the name given in the variable NAME
4.  To use the model follow the stages in [usage](#usage)


<a name="credits"></a>
## Credits
* Dataset: Prabhu, Vinay Uday. https://www.kaggle.com/c/Kannada-MNIST/overview