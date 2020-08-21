import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from PIL import Image

 
       


st.write(""" # Diabetes Detection """)
image=Image.open('F:/app/d.png')
st.image(image,use_column_width=True)
st.sidebar.title(" Welcome to Predictor ")
st.sidebar.info('App Contributor Avinash Bhat')    
 
d=pd.read_csv('dia.csv')
#st.subheader('data info')    
#st.dataframe(d)
#st.write(d.describe())
     
     
X=d.iloc[:,0:8].values
Y=d.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test=train_test_split(X,Y, test_size=0.355,random_state=101)

def user():
    Pregnancies=st.slider(' Pregnancies',0,17,3)
    Glucose=st.slider('Glucose',0,199,117)
    BloodPressure=st.slider('  BloodPressure',0,122,72)
    SkinThickness=st.slider(' SkinThickness',0,99,3)
    Insulin=st.slider(' Insulin',0.0,846.0,30.5)
    BMI=st.slider(' BMI',0.0,67.1,32.0)
    DPF=st.slider(' DiabetesPedigreeFunction',0.078,2.42,0.3725)
    Age=st.slider(' Age',0,81,29)
    user={ 'Pregnancies':Pregnancies,
           'Glucose':Glucose,
           ' BloodPressure': BloodPressure,
           'SkinThickness':SkinThickness,
           'Insulin':Insulin,
           'BMI':BMI,
           'DiabetesPedigreeFunction':DPF,
           'Age':Age
        
        
        
         }
    f=pd.DataFrame(user,index=[0])  
    return f
user_input =user()
#st.subheader('"user input"')
#st.write(user_input)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(x_train,y_train)
p=rfc.predict(x_test)
pre = rfc.predict(user_input)

if pre==1:
     st.write(""""You are Suffring from Diabetes""")
else:
    st.write("""You are not Suffring from Diabetes""")
st.sidebar.success('https://github.com/Avinashbhat1199')
 