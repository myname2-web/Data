import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
@st.cache_data

# Daten laden
df = pd.read_csv("train.csv")

st.title("Titanic : binary classification")
st.sidebar.title("Table of contents")
pages = ["Exploration", "DataVisualisation", "Modelling"]
page = st.sidebar.radio("Go to", pages)

if page == pages[0]:
    st.write("### Presentation of Data")
    st.dataframe(df.head(10))
    st.write(df.shape)
    st.dataframe(df.describe())

    if st.checkbox("Show NA"):
        st.dataframe(df.isna().sum())

        
if page == pages[1]:
    st.write("### DataVisualization")

    fig = plt.figure()
    sns.countplot(x="Survived", data=df)
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(x="Sex", data=df)
    plt.title("Distribution of the passengers gender")
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(x="Pclass", data=df)
    plt.title("Distribution of the passengers class")
    st.pyplot(fig)

    # Age - sicher konvertieren
    age_numeric = pd.to_numeric(df['Age'], errors='coerce')
    fig = plt.figure()
    sns.histplot(age_numeric.dropna(), bins=30)
    plt.title("Distribution of the passengers age")
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(x="Survived", hue="Sex", data=df)
    st.pyplot(fig)

    fig = sns.catplot(x="Pclass", y="Survived", data=df, kind="point")
    st.pyplot(fig)

    # Age vs Survival - nur mit numerischen Daten
    df_clean = df.copy()
    df_clean['Age'] = pd.to_numeric(df_clean['Age'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Age'])
    
    fig = sns.lmplot(x="Age", y="Survived", hue="Pclass", data=df_clean)
    st.pyplot(fig)

    # Korrelation - nur numerische Spalten
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots()
    sns.heatmap(numeric_df.corr(), annot=True, ax=ax)
    st.pyplot(fig)


if page == pages[2]:
    st.write("### Modelling")

    # Nur Spalten löschen, die tatsächlich existieren
    columns_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    df_model = df.drop(existing_columns, axis=1)
    
    st.write(f"Dropped columns: {existing_columns}")
    st.write(f"Remaining columns: {list(df_model.columns)}")
    
    y = df_model["Survived"]
    
    # Prüfe welche Spalten existieren
    available_cat = [col for col in ["Pclass", "Sex", "Embarked"] if col in df_model.columns]
    available_num = [col for col in ["Age", "Fare", "SibSp", "Parch"] if col in df_model.columns]
    
    X_cat = df_model[available_cat] if available_cat else pd.DataFrame()
    X_num = df_model[available_num] if available_num else pd.DataFrame()

    # Numerische Spalten sicher konvertieren
    if not X_num.empty:
        for col in X_num.columns:
            X_num[col] = pd.to_numeric(X_num[col], errors='coerce')

    # Fehlende Werte füllen
    if not X_cat.empty:
        for col in X_cat.columns:
            X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
    
    if not X_num.empty:
        for col in X_num.columns:
            X_num[col] = X_num[col].fillna(X_num[col].median())

    # Kategorische Variablen kodieren
    if not X_cat.empty:
        X_cat_encoded = pd.get_dummies(X_cat, columns=X_cat.columns)
        st.write(f"Categorical features shape: {X_cat_encoded.shape}")
        st.dataframe(X_cat_encoded.head())
    else:
        X_cat_encoded = pd.DataFrame()
        st.write("No categorical features found")
    
    if not X_num.empty:
        st.write(f"Numerical features shape: {X_num.shape}")
        st.dataframe(X_num.head())
    else:
        st.write("No numerical features found")
    
    # HIER: X erstellen durch Kombination aller Features
    X = pd.concat([X_cat_encoded, X_num], axis=1)
    st.write(f"Combined features shape: {X.shape}")
    
    # Machine Learning hinzufügen
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import confusion_matrix
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    # Modell-Auswahl
    choice = ['Random Forest', 'SVC', 'Logistic Regression']
    option = st.selectbox('Choice of the model', choice)
    st.write('The chosen model is:', option)
    
    # Modell erstellen
    if option == 'Random Forest':
        clf = RandomForestClassifier(random_state=123)
    elif option == 'SVC':
        clf = SVC(random_state=123)
    elif option == 'Logistic Regression':
        clf = LogisticRegression(random_state=123)
    
    # Modell trainieren
    clf.fit(X_train, y_train)
    
    # Ergebnisse anzeigen
    display = st.radio('What do you want to show?', ('Accuracy', 'Confusion matrix'))
    
    if display == 'Accuracy':
        accuracy = clf.score(X_test, y_test)
        st.write(f"Accuracy: {accuracy:.3f}")
    elif display == 'Confusion matrix':
        cm = confusion_matrix(y_test, clf.predict(X_test))
        st.write("Confusion Matrix:")
        st.dataframe(pd.DataFrame(cm, 
                                 columns=['Predicted: Died', 'Predicted: Survived'],
                                 index=['Actual: Died', 'Actual: Survived']))