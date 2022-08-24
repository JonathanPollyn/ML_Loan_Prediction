import pickle
import streamlit as st

import numpy as np
import pandas as pd
import seaborn as sns 

from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import numpy as np
import plotly.express as px
import io 

# Add Option Menu to Sidebar
with st.sidebar:
    choose = option_menu("App Gallery", ["About", "Data Exploration", "Predict a Loan", "Contact"],
                         icons=['house', 'activity', 'book','person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#000000"},
        "icon": {"color": "orange", "font-size": "15px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )


from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

loan_dataset =pd.read_csv(r'loan.csv')

loan_dataset = loan_dataset.dropna()

loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)

loan_dataset = loan_dataset.replace(to_replace='3+', value=4)

loan_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)

X = loan_dataset[['Gender', 'Married', 'ApplicantIncome', 'LoanAmount', 'Credit_History']]
y = loan_dataset.Loan_Status

from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size = 0.2, random_state = 10)

from sklearn.ensemble import RandomForestClassifier 
model = RandomForestClassifier(max_depth=4, random_state = 10) 
model.fit(x_train, y_train)

from sklearn.metrics import accuracy_score
pred_cv = model.predict(x_cv)
accuracy_score(y_cv,pred_cv)

pred_train = model.predict(x_train)
accuracy_score(y_train,pred_train)

pickle_out = open(r"C:\Python_GUI\Loan_Prediction_Machine\classifier.pkl", mode = "wb") 
pickle.dump(model, pickle_out) 
pickle_out.close()


 
# loading the trained model
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)
 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History):   
 
    # Pre-processing user input    
    if Gender == "Male":
        Gender = 0
    else:
        Gender = 1
 
    if Married == "Single":
        Married = 0
    else:
        Married = 1
 
    if Credit_History == "Outstanding Debts":
        Credit_History = 0
    else:
        Credit_History = 1  

    # if ApplicantIncome <= 20000.00 & ApplicantIncome >= 250000.00:
    #     ApplicantIncome = 0
    # else:
    #     ApplicantIncome = 1
 
    LoanAmount = LoanAmount / 1000
 
    # Making predictions 
    prediction = classifier.predict( 
        [[Gender, Married, ApplicantIncome, LoanAmount, Credit_History]])
     
    if prediction == 0:
        pred = 'Rejected'
    else:
        pred = 'Approved'
    return pred
      
  
# this is the main function that is define the webpage  
def main():       
    st.write('For any issues regarding the functionality of the application, please contact Jonathan Pollyn')


    # Create App Pages
# About page
if choose == "About":
   st.title('Loan prediction system')
   st.write(('This project aims to create a "Data science product" such as a "Loan prediction system" for the bank. The product will automate the loan approval process as it focuses on reducing the manual time and effort involved in the loan approval process. This project evaluation measure or acceptance criteria are accurate and reduce false positives. Approving a loan for an applicant without eligibility and the capability to repay the loan will pose a severe challenge for the bank. The main acceptance criteria for this data science project are to increase accuracy and reduce the false positive rate. In addition to the final product, the project includes a user manual to work on this system. This manual is helpful for the users to understand the product thoroughly, how it works, and eligible input and expected output from the data product'))
  
    # Photo editing page
elif choose == 'Data Exploration':
    st.write(loan_dataset.head())
    st.write(loan_dataset.describe())

elif choose == 'Predict a Loan':
    # this is the main function that is define the webpage  
    def main():       
    # front end elements of the web page 
        html_temp = """ 
        <div style ="background-color:blue;padding:13px"> 
        <h1 style ="color:black;text-align:center;">Loan Prediction system</h1> 
        </div> 
        """
      
        # display the front end aspect
        st.markdown(html_temp, unsafe_allow_html = True) 
      
        # following lines create boxes in which user can enter data required to make prediction 
        Gender = st.selectbox('Gender',("Male","Female"))
        Married = st.selectbox('Marital Status',("Single","Married")) 
        ApplicantIncome = st.number_input("Applicants monthly income") 
        LoanAmount = st.number_input("Total loan amount")
        Credit_History = st.selectbox('Credit_History',("Outstanding Debts","No Outstanding Debts"))
        result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
        if st.button("Predict"): 
            result = prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History) 
            st.success('Your loan is {}'.format(result))
            print(LoanAmount)
elif choose == 'Contact':
    st.markdown(""" <style> .font {
                font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
                </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Contact Form</p>', unsafe_allow_html=True)
    with st.form(key='columns_in_form2',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
    #st.write('Please help us improve!')
        Name=st.text_input(label='Please Enter Your Name') #Collect user feedback
        Email=st.text_input(label='Please Enter Email') #Collect user feedback
        Message=st.text_input(label='Please Enter Your Message') #Collect user feedback
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.write('Thank you for the contact and a representative will get in touch with you within 24 hours')
     
if __name__=='__main__': 
    main()