# Importing Libraries

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import warnings
warnings.filterwarnings('ignore')
matplotlib.use('tkagg')
from PIL import Image

# Streamlit Containers

header = st.container()
dataset = st.container()
plot = st.container()
kendall = st.container()
chi2 = st.container()
classification = st.container()

# Title

with header:
    st.title("Welcome to a User Friendly Application")
    img = Image.open("prediction.jpg")
    st.image(img, width = 500)

    st.markdown("### In this, We are going to predict a Disease by using Symptoms.")

# Dataset

with dataset:
    st.title("DataSet")
    st.markdown("#### Description")
    st.info("* The Dataset was extracted from Kaggle. This dataset has 132 parameters on which 3 different types of diseases can be predicted.")
    st.text("To load the DataSet, Please Click the 'View DataSet' button given below...")
    if st.button("View DataSet"):
        data = pd.read_csv(r"C:\Users\DELL\OneDrive\Documents\Python\Cleanical Decision Support Sytem Project\sampletraining.csv")
        st.write(data)

    if st.button("Details of Dataset"):
        data = pd.read_csv(r"C:\Users\DELL\OneDrive\Documents\Python\Cleanical Decision Support Sytem Project\sampletraining.csv")
        shape = data.shape

        st.write("The shape of the dataset is:", shape)
        describe = data.describe()
        st.write("To show a quick statistics summary of the data:", describe)


# Data Visualization

with plot:
    st.title("Data Visualization using PyPlot Diagram")
    if st.button("PyPlot"):
        data = pd.read_csv(r"C:\Users\DELL\OneDrive\Documents\Python\Cleanical Decision Support Sytem Project\sampletraining.csv")
        disease_counts = data["prognosis"].value_counts()
        temp_df = pd.DataFrame({
            "Disease": disease_counts.index,
            "Counts": disease_counts.values})
        plt.figure(figsize=(8,6))
        sns.barplot(x="Disease", y="Counts", data=temp_df)
        st.pyplot(plt)
        plt.close()

# Feature Selection using Kendall

with kendall:
    st.title("Feature Selection")
    st.markdown(" **Kendall's Coefficient Correlation**")
    if st.button("Kendall"):
        data = pd.read_csv(r"C:\Users\DELL\OneDrive\Documents\Python\Cleanical Decision Support Sytem Project\sampletraining.csv")
        dff = data.drop(["prognosis"], axis=1)
        corr_matrix = dff.corr(method='kendall')

        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))

        # Find index of feature columns with correlation greater than 0
        to_view = [column for column in upper.columns if any(upper[column] > 0)]
        cr = corr_matrix.style.background_gradient(cmap='coolwarm')
        cr

        high_corr_var = np.where(corr_matrix > 0)
        high_corr_var = [(corr_matrix.columns[x], corr_matrix.columns[y]) for x, y in zip(*high_corr_var) if
                         x != y and x < y]
        print("Lenght:", len(high_corr_var))
        high_corr_var

    if st.button("Kendall Featurs"):
        st.info("From the 132 Symptoms given, 19 symptoms are selected which are highly correlated to each other. Other poorly correlated features are eliminated.")
        st.write("They are")
        st.write("""
        - Itching
        - Fatigue
        - Weight Loss 
        - Yellowish Skin
        - Dark Urine
        - Abdominal Pain
        - Chills 
        - Skin Rash
        - Joint Pain
        - Headache
        - Nausea
        - Loss of Appetite
        - Pain behind the eyes
        - Back Pain
        - Malaise
        - Muscle Pain
        - Red spots over body
        - Sweating 
        - Diarrhoea """)

# Features Selected using Chi-Square

with chi2:
    st.markdown(" **Chi-Square Test**")
    if st.button("Chi-Square Features"):
        st.info(
            "From the 132 Symptoms given, 21 symptoms are selected which are highly correlated to each other. Other poorly correlated features are eliminated.")
        st.write("They are")
        st.write("""
               - Itching
               - Fatigue
               - Weight Loss 
               - Yellowish Skin
               - Dark Urine
               - Abdominal Pain
               - Chills 
               - Skin Rash
               - Joint Pain
               - Headache
               - Nausea
               - Loss of Appetite
               - Pain behind the eyes
               - Back Pain
               - Malaise
               - Muscle Pain
               - Red spots over body
               - Sweating 
               - Diarrhoea 
               - Vomiting 
               - High Fever""")



# Classification

with classification:
    st.title("Classification using CNN")
    data = pd.read_csv(r"C:\Users\DELL\OneDrive\Documents\Python\Cleanical Decision Support Sytem Project\sampletraining.csv")

    # Data With Correlated symptoms using Kendall and Chi-Square

    overall_data =data[['itching',  'fatigue', 'weight_loss', 'yellowish_skin','dark_urine', 'abdominal_pain', 'chills','skin_rash','joint_pain','headache','nausea','loss_of_appetite', 'pain_behind_the_eyes','back_pain','malaise','muscle_pain','red_spots_over_body','sweating','diarrhoea','vomiting','high_fever','prognosis']].copy()

    st.info(" The new dataset with Correlated symptoms using Kendall and Chi-Square")
    if st.button("Correlated Dataset"):
        st.write(overall_data)
    if st.button("CNN"):
        st.markdown(" **Training and Testing**")

        # Training and Testing data set

        training = overall_data
        testing = pd.read_csv(r"C:\Users\DELL\OneDrive\Documents\Python\Cleanical Decision Support Sytem Project\sample_testing.csv")
        st.write(f"No. of training examples: {training.shape[0]}")
        st.write(f"No. of testing examples: {testing.shape[0]}")
        X_train = training.drop('prognosis', axis=1)
        X_test = testing.drop('prognosis', axis=1)

        y_train = np.array(training['prognosis'])
        y_test = np.array(testing['prognosis'])

        y_train_enc = pd.get_dummies(y_train)
        y_test_enc = pd.get_dummies(y_test)

        # Building CNN Model


        model = Sequential()  # Initializing the CNN Model

        # adding first hidden layer with input layer. there is init parameter that represents how to initialize weights
        model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))

        # adding second hidden layer
        model.add(Dense(32, activation='relu'))

        # adding third hidden layer
        model.add(Dense(16, activation='relu'))

        # adding last layer
        model.add(Dense(y_train_enc.shape[1], activation='softmax'))

        # Summarizing the CNN Model

        print("Summarizing the CNN Model")
        model.summary()

        # Compile model using accuracy to measure model performance
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Training the CNN Model
        model.fit(X_test, y_test_enc, epochs=30)

        # A convolutional neural network can be evaluated using the 'evaluate' method

        model.evaluate(X_test, y_test_enc)


        Xnew = training.drop('prognosis', axis=1)
        ynew = training['prognosis']

        from keras.optimizers import Adam

        def create_model(learning_rate, activation):
            model2 = Sequential()
            my_opt = Adam(lr=learning_rate)
            model2.add(Dense(64, activation=activation, input_shape=(X_train.shape[1],)))

            model2.add(Dense(y_train_enc.shape[1], activation='softmax'))
            model2.compile(optimizer=my_opt, loss='categorical_crossentropy', metrics=['accuracy'])
            return model2


        from keras.wrappers.scikit_learn import KerasClassifier

        modelnew = KerasClassifier(build_fn=create_model, epochs=30, batch_size=100, validation_split=0.3)

        from sklearn.model_selection import RandomizedSearchCV

        params = {'activation': ['relu', 'tanh'], 'batch_size': [32, 128, 256],
                            'epochs': [10], 'learning_rate': [0.1, 0.01, 0.001]}

        random_search = RandomizedSearchCV(modelnew, param_distributions=params, cv=5)
        r = random_search.fit(Xnew, ynew)

        history = random_search.best_estimator_.fit(Xnew, ynew)

        Xtestnew = testing.drop('prognosis', axis=1)
        ytestnew = testing['prognosis']

        random_search.best_estimator_.score(Xtestnew, ytestnew)

        st.markdown(" **Accuracy**")
        history.history["accuracy"]

        import matplotlib.pyplot as plt

        # Model Accuracy

        plt.plot(history.history["accuracy"])
        plt.title("Model accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy score")
        plt.show()

        # Model Loss

        plt.plot(history.history["loss"])
        plt.title("Model loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

        # prediction

        st.markdown(" **Prediction of the diseases are:**")
        pred = random_search.best_estimator_.predict(X_test)
        pred

        # Confusion Matrix

        st.markdown(" **Confusion Matrix for CNN Classifier using Test Data**")
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report

        preds = r.predict(X_test)
        cf_matrix = confusion_matrix(y_test, preds)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cf_matrix, annot=True)
        plt.title("Confusion Matrix for CNN Classifier on Test Data")
        plt.show()

        # Classification Report using Confusion Matrix
        confusion_matrix_LightGBM = confusion_matrix(y_test, preds)
        from sklearn.metrics import classification_report

        classification_report(y_test, preds)
        st.text('Model Report:\n ' + classification_report(y_test, preds) )














