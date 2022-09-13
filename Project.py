import streamlit as st
from streamlit_option_menu import option_menu
# pre-processing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# machine learning
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from PIL import Image
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , classification_report, confusion_matrix


# icon 
st.set_page_config(page_title="Brain Stroke Prediction", page_icon=":brain:")

# Title
st.title("😎 Brain Stroke Prediction App 😎")
st.markdown("---")

#Lodainng the dataset
d_data= pd.read_csv("brain_stroke.csv")

# Main Menu
with st.sidebar:
    options = option_menu(
        menu_title="Menu", 
        options=["Introduction", "Exploratory Data Analysis", "Data Visualization", "Scaling the Data", "Model Building"]
    )

# # Side bar
# options=st.sidebar.radio("MAIN MANU",
#     ["Introduction","Exploratory Data Analysis","Data Visualization","Scaling the Data","Model Building"]
# )

# Content
def Content(options):
    if options == "Introduction":
        Introduction()
    elif options == "Exploratory Data Analysis":
        EDA()
    elif options == "Data Visualization":
        Visualizations()
    elif options == "Scaling the Data":
        Scaling_Data()
    elif options == "Model Building":
        Model_Building()

d_data_copy = d_data.copy(deep = True)


# Dataset Description
def Introduction():
    st.header("😊 Introduction 😊")
    st.write("""
Story of dataset:

A stroke is a medical condition in which poor blood flow to the brain causes cell death. There are two main types of stroke: ischemic, due to lack of blood flow, and hemorrhagic, due to bleeding. Both cause parts of the brain to stop functioning properly. Signs and symptoms of a stroke may include an inability to move or feel on one side of the body, problems understanding or speaking, dizziness, or loss of vision to one side. Signs and symptoms often appear soon after the stroke has occurred. If symptoms last less than one or two hours, the stroke is a transient ischemic attack (TIA), also called a mini-stroke. A hemorrhagic stroke may also be associated with a severe headache. The symptoms of a stroke can be permanent. Long-term complications may include pneumonia and loss of bladder control.

The main risk factor for stroke is high blood pressure. Other risk factors include high blood cholesterol, tobacco smoking, obesity, diabetes mellitus, a previous TIA, end-stage kidney disease, and atrial fibrillation. An ischemic stroke is typically caused by blockage of a blood vessel, though there are also less common causes. A hemorrhagic stroke is caused by either bleeding directly into the brain or into the space between the brain's membranes. Bleeding may occur due to a ruptured brain aneurysm. Diagnosis is typically based on a physical exam and is supported by medical imaging such as a CT scan or MRI scan. A CT scan can rule out bleeding, but may not necessarily rule out ischemia, which early on typically does not show up on a CT scan. Other tests such as an electrocardiogram (ECG) and blood tests are done to determine risk factors and rule out other possible causes. Low blood sugar may cause similar symptoms.

Prevention includes decreasing risk factors, surgery to open up the arteries to the brain in those with problematic carotid narrowing, and warfarin in people with atrial fibrillation. Aspirin or statins may be recommended by physicians for prevention. A stroke or TIA often requires emergency care. An ischemic stroke, if detected within three to four and half hours, may be treatable with a medication that can break down the clot. Some hemorrhagic strokes benefit from surgery. Treatment to attempt recovery of lost function is called stroke rehabilitation, and ideally takes place in a stroke unit; however, these are not available in much of the world.

Attribute Information
1) gender: "Male", "Female" or "Other"
2) age: age of the patient
3) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
4) heart disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease 5) ever-married: "No" or "Yes"
6) worktype: "children", "Govtjov", "Neverworked", "Private" or "Self-employed" 7) Residencetype: "Rural" or "Urban"
8) avgglucoselevel: average glucose level in blood
9) bmi: body mass index
10) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
11) stroke: 1 if the patient had a stroke or 0 if not

*Note: "Unknown" in smoking_status means that the information is unavailable for this patient

Real data sources:
Data
    """)
    image = Image.open('image.webp')
    st.image(image, caption='Brain Stroke Prediction', use_column_width=True)



# Exploratory Data Analysis
def EDA():
    st.header("🕵️ Exploratory Data Analysis 🕵️")
    
    st.write("### ***Enter the number of rows to view***")
    rows = st.number_input("", min_value=0,value=5)
    if rows > 0:
        st.table(d_data.head(rows))
    
    st.write("---")
    st.subheader("Number of rows and columns")
    st.write(f'Rows: {d_data.shape[0]}')
    st.write(f'Columns: {d_data.shape[1]}')
    
    st.write("---")
    st.subheader("Columns of the Dataset")
    st.table(d_data.columns)
    
    st.write("---")
    st.subheader("Summary Statistics")
    st.write(d_data.describe().T)
    
    st.write("---")
    st.subheader("Information of the Dataset")
    st.write(d_data.info())
    
    st.write("---")
    st.subheader("Unique Values")
    st.table(d_data.nunique())
    
    st.write("---")
    st.subheader("Indexing Values")
    st.text(d_data.index)
    
    st.write("---")
    st.subheader("Missing/Null Values in the Dataset")
    st.table(d_data.isnull().sum())    
    
    st.write("In this particular dataset all the missing values are 0. So, we don't need to do any imputation.")
    
    
# Data Visualization
def box_plots():
    st.write("The following are the box plots of the dataset before removing the null values")
    fig1=px.box(d_data, x="age",title="boxplot of age groups")
    st.plotly_chart(fig1)
    
    fig2=px.box(d_data, x="avg_glucose_level",title="boxplot of Average group level")
    st.plotly_chart(fig2)
    
    fig3=px.box(d_data, x="bmi",title="body mass index of groups")
    st.plotly_chart(fig3)
    
    st.write("The box plots of all columns shows some outliers but the some columns shows so many outliers in the dataset which is not good for the authenticity of the dataset and hence we will remove the outliers")
    if st.checkbox("Remove the Outliers from Columns"):
        st.write("The outliers in the avg_glucose_level column have been removed")
        d_data_copy = d_data.copy(deep = True)
        d_data_copy = d_data_copy[d_data_copy['avg_glucose_level']<142]
        fig4=px.box(d_data_copy,x="avg_glucose_level",title="boxplot of Average group level")
        st.plotly_chart(fig4)    
        
        st.write("The outliers in the bmi column have been removed")
        d_data_copy = d_data.copy(deep = True)
        d_data_copy = d_data_copy[d_data_copy['bmi']<45]
        fig4=px.box(d_data_copy,x="bmi",title="boxplot of Average group level")
        st.plotly_chart(fig4)

def histograms():
    st.subheader("Distribution Plots After Removing Null Values")
    st.write("The following are the distribution plots of the dataset after removing the null values")
    fig1=px.histogram(d_data,x="heart_disease", color="stroke",title="Heart Disease vs Brain Stroke", barmode="group", text_auto="percent+label")
    st.plotly_chart(fig1)
    fig2=px.histogram(d_data,x="Residence_type", color="stroke", title="Poeples from Urban & Rural", barmode="group", text_auto="percent+label")
    st.plotly_chart(fig2)
    fig3=px.histogram(d_data,x="gender", color="hypertension", title="Hypertension of Males and females", barmode="group", text_auto="percent+label")
    st.plotly_chart(fig3)
    fig4=px.histogram(d_data, x="ever_married", color="stroke", title="Married & Unmarried having stroke", barmode="group", text_auto="percent+label")
    st.plotly_chart(fig4)
    fig5=px.histogram(d_data,x="gender", color="stroke",title="Males and females Stroke", barmode="group", text_auto="percent+label")
    st.plotly_chart(fig5)
    fig6=px.histogram(d_data,x="work_type", color="stroke",title="Different Workers having Brain Stroke", barmode="group", text_auto="percent+label")
    st.plotly_chart(fig6)
    fig7=px.histogram(d_data, x="smoking_status", color="stroke", title="Smokers having Brain Stroke", barmode="group", text_auto="percent+label")
    st.plotly_chart(fig7)
    fig8=px.histogram(d_data,x="age", color="stroke",title="count of age groups having stroke",  text_auto="percent+label")
    st.plotly_chart(fig8)
    st.write("from above visualizations we compared all almost columns with stroke")


def bar_plot():
    plot_bar = px.bar(d_data, x="work_type", y="age", color="stroke", title="Different age group worker having stroke",  text_auto="label")
    st.plotly_chart(plot_bar)
    st.write("Hence, Private workers are more prone to stroke")
    
def scatter_plot():
    plot_scatter = px.scatter(d_data, x="age", y="bmi", color="stroke", title="scatter plot of age and avg_glucose_level")
    st.plotly_chart(plot_scatter)
    st.write("Hence, the people having stroke are more in the age group of 50-60 and the people having stroke are more in the bmi range of 20-40")
    
def area_plot():
    plot_scatter = px.area(d_data, x="age", y="avg_glucose_level", color="stroke", title="area plot of age and avg_glucose_level")
    st.plotly_chart(plot_scatter)
    st.write("Hence, the people having stroke are more in the age group of 40-80 and more peoples are in the avg_glucose_level range of 100-300")
    
def countplots():
    co_plot=px.bar(d_data['bmi'].value_counts(),title="Count Plot of Outcome")
    st.plotly_chart(co_plot)
    st.write("The above count plot shows that the dataset contains max bmi 42")
    
    
def piechart():
    pie=px.pie(d_data, names="stroke",title="Pie Chart of Stroke")
    st.plotly_chart(pie)
    st.write("The above pie chart shows that 5% of peoples haing brain stroke")

def correlation():
    st.write("The following is the correlation matrix of the dataset")
    d_data["ever_married"] = LabelEncoder().fit_transform(d_data["ever_married"])
    corr=d_data.corr()
    h_map=sns.heatmap(corr,annot=True)
    st.pyplot(h_map.figure)
    st.write("The above correlation matrix shows that the columns stroke and age are more correlated with each other, where age, bmi, ever_married are having higher correlation.")

def Visualizations():
    st.header("📈📉 Data Visualization 📉📈")
    st.markdown("---")
    st.subheader("Box Plots Before Removing Null Values")
    box_plots()
    
    st.markdown("---")
    st.subheader("Histograms")
    histograms()
    
    st.markdown("---")
    st.subheader("Countplots")
    countplots()
    
    st.markdown("---")
    st.subheader("Pie Chart")
    piechart()
    
    st.markdown("---")
    st.subheader("Bar Plot")
    bar_plot()
    
    st.markdown("---")
    st.subheader("Scatter Plot")
    scatter_plot()
    
    st.markdown("---")
    st.subheader("Area Plot")
    area_plot()
    
    st.markdown("---")
    st.subheader("Heat Map")
    correlation()

# Pre-Processsing
def Scaling_Data():
    st.header("🧑‍💻 Data Preprocessing 🧑‍💻")
    # label encoding
    d_data['ever_married'] = [ 0 if i !='Yes' else 1 for i in d_data['ever_married'] ]
    d_data['gender'] = [0 if i != 'Female' else 1 for i in d_data['gender']]
    d_data["work_type"] = LabelEncoder().fit_transform(d_data["work_type"])
    d_data["Residence_type"] = LabelEncoder().fit_transform(d_data["Residence_type"])
    d_data["smoking_status"] = LabelEncoder().fit_transform(d_data["smoking_status"])
    
    st.write("### Enter the number of rows to view")
    rows = st.number_input("", min_value=0,value=5)
    if rows > 0:
        st.table(d_data.head(rows))
        st.write("That’s how our dataset will be looking like when it is encoded.")
    
    # data division
    X = d_data.iloc[:,0:10]
    y = d_data.iloc[:,10]
    
    # Standar Scaling
    data_copy = d_data.copy(deep= True)
    X = StandardScaler().fit_transform(X)
    X = pd.DataFrame(X, columns = data_copy.columns[0:10])
    
    st.write("---")
    st.write("### Enter the number of rows to view")
    rows = st.number_input("", min_value=0,value=5, key=21)
    if rows > 0:
        st.table(X.head(rows))
        st.write("That's how our dataset will be looking like when it is scaled down or we can see every value now is on the same scale which will help our ML model to give a better result.")


# Selecting & Building Model
def Model_Building():
    st.header("🏨 Model Building 🏨")
    if st.checkbox("Input Data and Output Data"):
        # label encoding
        d_data['ever_married'] = [ 0 if i !='Yes' else 1 for i in d_data['ever_married'] ]
        d_data['gender'] = [0 if i != 'Female' else 1 for i in d_data['gender']]
        d_data["work_type"] = LabelEncoder().fit_transform(d_data["work_type"])
        d_data["Residence_type"] = LabelEncoder().fit_transform(d_data["Residence_type"])
        d_data["smoking_status"] = LabelEncoder().fit_transform(d_data["smoking_status"])
        # data division
        X = d_data.iloc[:,0:10]
        y = d_data.iloc[:,10]
        # Standar Scaling
        data_copy = d_data.copy(deep= True)
        X = StandardScaler().fit_transform(X)
        X = pd.DataFrame(X, columns = data_copy.columns[0:10])
        
        
        st.write("### Enter the number of rows to view")
        rows = st.number_input("", min_value=0,value=5)
        if rows > 0:
            st.table(X.head(rows))
            st.table(y.head(rows))
        
        
        st.write("---")
        st.subheader("Splitting the Data")
        train_size=st.selectbox("Select Train Size",[0.7,0.8,0.9])
        random_state=st.selectbox("Select Random State",[0,1,21,33,42])
        X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=train_size,random_state=random_state)
        st.write("The data has been splitted into train and test data")
        st.write("The shape of X_train is",X_train.shape)
        st.write("The shape of X_test is",X_test.shape)
        st.write("The shape of y_train is",y_train.shape)
        st.write("The shape of y_test is",y_test.shape)
        
        st.write("---")    
        st.subheader("Model Selection")
        classifier_name = st.selectbox("Select Classifier",("SVM", "Logistic Regression", "KNN", "ANN", "Random Forest","Desicion Tree"))
        def add_parameter(classifier_name):
            params=dict()
            if classifier_name=="SVM":
                st.sidebar.subheader("Model Parameters")
                C=st.sidebar.slider("C",0.01,10.0)
                params["C"]=C
            if classifier_name=="Logistic Regression":
                st.sidebar.subheader("Model Parameters")
                c=st.sidebar.slider("c",0.01,10.0)
                params["c"]=c
            elif classifier_name=="KNN":
                st.sidebar.subheader("Model Parameters")
                K=st.sidebar.slider("K",1,20)
                params["K"]=K
            elif classifier_name=="ANN":
                st.sidebar.subheader("Model Parameters")
                max_iter = st.sidebar.slider('Max_iter', 50, 500)
                params['max_iter'] = max_iter
                alpha = st.sidebar.slider('Alpha', 0.0001, 10.0)
                params['alpha'] = alpha
            elif classifier_name=="Random Forest":
                st.sidebar.subheader("Model Parameters")
                max_depth=st.sidebar.slider("Max Depth",2,15)
                params['max_depth']=max_depth
                n_estimators=st.sidebar.slider("Number of Estimators",1,200)
                params['n_estimators']=n_estimators
            elif classifier_name=="Desicion Tree":
                st.sidebar.subheader("Model Parameters")
                max_depth=st.sidebar.slider("Max Depth",2,15)
                params['max_depth']=max_depth
                criterion=st.sidebar.selectbox("Select Criterion",["gini","entropy"])
                params['criterion']=criterion
            return params
        
        params=add_parameter(classifier_name)
            
        def get_classifier(classifier_name,params):
            classifier=None
            if classifier_name=="SVM":
                classifier=SVC(C=params["C"])
            elif classifier_name=="Logistic Regression":
                classifier=LogisticRegression(C=params["c"], random_state=random_state)
            elif classifier_name=="KNN":
                classifier=KNeighborsClassifier(n_neighbors=params["K"])
            elif classifier_name == "ANN":
                classifier=MLPClassifier(max_iter=params['max_iter'], alpha = params['alpha'], random_state=random_state)    
            elif classifier_name=="Random Forest":
                classifier=RandomForestClassifier(n_estimators=params['n_estimators'],max_depth=params['max_depth'],random_state=random_state)
            elif classifier_name=="Desicion Tree":
                classifier=DecisionTreeClassifier(criterion=params['criterion'],max_depth=params['max_depth'])
            return classifier
            
        classifier=get_classifier(classifier_name,params)
        classifier.fit(X_train,y_train)
        y_pred=classifier.predict(X_test)
        st.write("The accuracy of the model is", accuracy_score(y_test,y_pred))
        st.write("The confusion matrix of the model is")
        st.table(confusion_matrix(y_test,y_pred))
        st.write("Classification Report: ")
        st.write(classification_report(y_test,y_pred))

# Calling whole Data
Content(options)


