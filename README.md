# AI & CogSci Midterm Project

## Description

This project focuses on detecting positive, neutral, or negative feelings based on  text. These feelings include but are not limited to anger, fear, happiness, sadness, and surprise. We design and implement the neural network that imitates human ability of text-based empathy of emotions. To train our neural network, we use existing datasets ISEAR and TWEETS, containing information of texts and their corresponding emotions. The goal of this neural network is to predict human emotion based on the input sentence.

## Get Started

#### Package Dependency
Please install all the package in the [requirements](requirements.txt), including;
```bash
pip install pandas pandas_access nltk numpy keras tensorflow
```

Copy the following lines into a python file and download the content.

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

#### Structures

Dataset used in this project is located in the following places:
* [ISEAR](Dataset/ISEAR)
* [Tweets](Dataset/Tweets.csv)

All the source code is placed in [here](src):
* [Preprocess](src/preprocessing.py) provides function to preprocessing the text data before vectorizing them into number vectors. 
* [Model](src/model_builder.py) includes all the layer 
* [MDB Reader](src/mdb_sav_reader.py) provides a function to decode `.mdb` file.
* [Embedding](src/embedding.py) helps to create the embedding layer in neural network
* [SVM](src/SVM.py) helps to create svm classification methods

[Glove](glove) is a pretrained embedding layer for text training.  

[Doc](doc) includes all the designed documentation inside

[Ipynb](ipynb) contains all the jupyter notebook that shows a demo of how our neural network performs 

#### Run

Sample code for each dataset is placed in [here](src). All the scripts for two datasets is runable by command:

```bash
python <filename>
```

##### I. [ISEAR](src/ISEAR.py)
This file provides detailed implementation of how to using this dataset to train the neural network. With some adjustment, the best performance is around 60% accuracy. In the file, we use 7 categories of emotions for classifications. 

##### II. [TWEETS](src/Tweets.py)
This file provides implementations for Tweets dataset, where the number of categories can be 13 or 3, which is adjustable by changing the constant `SIMPLE_CATEGORY`. Other constants is used for different meaning as indicating inside the file. 

#### Further work
Further work can implement by importing scripts in [src](src) folder.

## Reference
* [LSTM Intro](https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17)
* [How to read mdb file?](https://stackoverflow.com/questions/3620539/how-to-deal-with-mdb-access-files-with-python)
* [Text Preprocessing](https://medium.com/analytics-vidhya/text-preprocessing-for-nlp-natural-language-processing-beginners-to-master-fd82dfecf95)
* [GloVe](https://nlp.stanford.edu/projects/glove/)
* [Tweets Database](https://data.world/crowdflower/sentiment-analysis-in-text)
* [ISEAR Database](https://www.unige.ch/cisa/research/materials-and-online-research/research-material/)
