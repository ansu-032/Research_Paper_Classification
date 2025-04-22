# Research_Paper_Classification

This project attempts to classify Resurch paper documents into Publishable and Non-Publishable based on a deep learning solution. It provides an end-to-end pipeline from raw PDF to a trained model for classification. The main aim is to help automate categorizing research or scholarly papers based on their content so that editorial or review processes could be automated.

The input data for this project are PDF files in two folders called "Publishable" and "Non-Publishable." A specific Python function is employed to recursively upload these folders and extract text data from all the PDFs using the PyPDF2 library. The extracted text is saved along with its label to allow the model to learn patterns that distinguish between the two classes.

After the text data is extracted, it is translated into a numerical form using CountVectorizer from scikit-learn. This is necessary so that the text can be input into a model of a neural network. The data is split into test and training sets to gauge how the model would act in real-world scenarios.

A basic and effective feedforward neural network is implemented through TensorFlow's Keras API. The model consists of dense layers with ReLU activation and dropout layers to avoid overfitting. The model is trained to classify whether a document can be published or not based on the content read in PDF documents. Once trained, the model is evaluated for accuracy and performance through the aid of parameters such as precision, recall, and F1-score.

The notebook also includes functionality to store the trained model in the .h5 format, and then load it again for prediction on new PDF documents. Additionally, plots are generated to observe the training of the model over time, and get an idea of its learning pattern.

