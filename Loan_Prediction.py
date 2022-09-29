import pickle
import streamlit as st

import numpy as np
import pandas as pd
import seaborn as sns 

import plotly.offline as pyo
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.express as px

from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import numpy as np
import plotly.express as px
import io 



# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data



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
    st.write('')

# Add Option Menu to Sidebar
# st.title('App information')
menu = ["Home","About", "login", "SignUp"]
choose = st.sidebar.selectbox("Menu", menu)


    # Create App Pages
if choose == "Home":
    st.title('Welcome')
    st.write("Thank you for visiting, my name is Jonathan Pollyn, and this web application is inspired by my capstone project for my master's degree program with Grand Canyon University. I currently work as a data engineer and am passionate about databases, Business Intelligence, and data science.")

    st.write(' I have developed enterprise-level Business Intelligence solutions for various industries, focusing on performance optimization, pattern recognition, efficient analysis of business processes, and interactive visualizations. As part of my program with GCU, I have been exposed to various learning tools that have strengthened my knowledge and helped me to understand the future of data science.')

    st.write('If you have any questions, do not hesitate to get in touch with me via email at j.pollyn@gmail.com')
# About page
elif choose == "About":
    st.title('Loan prediction system')
    st.write(('This project aims to create a "Data science product" such as a "Loan prediction system" for the bank. The product will automate the loan approval process as it focuses on reducing the manual time and effort involved in the loan approval process. This project evaluation measure or acceptance criteria are accurate and reduce false positives. Approving a loan for an applicant without eligibility and the capability to repay the loan will pose a severe challenge for the bank.'))
    st.write(('The main acceptance criteria for this data science project are to increase accuracy and reduce the false positive rate. In addition to the final product, the project includes a user manual to work on this system. This manual is helpful for the users to understand the product thoroughly, how it works, and eligible input and expected output from the data product.'))
    st.write(( 'The main focus of the loan prediction system is to predict whether an applicant can repay the loan amount. To predict that, it must process an applicantloan application. Machine Learning predictive model will process the application of an applicant. Data from the application will be passed as input to the model.'))
elif choose == "login" :
    st.subheader('Login')

    username = st.sidebar.text_input('User Name')
    password = st.sidebar.text_input('Password', type='password')

    if st.sidebar.checkbox('Login'):
        # if password == '12345':
        create_usertable()
        result = login_user(username, password)
        if result:
            
            st.success('Logged in as {}'.format(username))

            task = st.selectbox('Task', ['Predict a load Application' , 'Data Exploration','Contacts'])
            if task == 'Predict a load Application':
                st.subheader('Predict your loan eligibility')
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



            elif task == 'Data Exploration':
                st.subheader('Perform your analysis')
                st.title("Data Exploration for Loan Application")
                st.markdown('This section of the code is designated for performing various exploratory data analysis. You can upload your own file for the analysis. Also check below for more options of analysis')
                loan_file = st.file_uploader('Select Your Local Loan historical CSV (default provided)')
                if loan_file is not None:
                        loan_df = pd.read_csv(loan_file)
                        st.write(loan_dataset.head())
                        st.write(loan_dataset.describe())
                else:
                        loan_df= pd.read_csv('loan.csv')
                        st.write(loan_dataset.head())
                        st.write(loan_dataset.describe())
                
                        st.subheader('Plotly Histogram Chart')
                        selected_x_var = st.selectbox('What do want the x variable to be?', ['Gender','Self_Employed','Education'])
                        fig = px.histogram(loan_df[selected_x_var])
                        st.plotly_chart(fig)

                        st.subheader('Plotly Bar Chart')
                        selected_x_bar = st.selectbox('Select from the avaiable x variable: ', ['Married','Property_Area'])
                        selected_y_bar = st.selectbox('Select from the avaiable y variable', ['Dependents','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term'])

                        data = [go.Bar(
                        x=loan_df[selected_x_bar], 
                        y=loan_df[selected_y_bar]
                        )]
                        layout = go.Layout(
                        title="Bar chart for: {},{}".format(selected_x_bar,selected_y_bar)
                        )
                        fig = go.Figure(data=data, layout=layout)
                        st.plotly_chart(fig)

            elif task == 'Contacts':
                # st.subheader('User Contactss')
                # user_result = view_all_users()
                # clean_db = pd.DataFrame(user_result, columns=['Username', 'Password'])
                # st.dataframe(clean_db)
                st.write('Coming Soon')
        else:
            st.warning('Incorrect Username or Password')
elif choose == 'SignUp':
        new_user = st.text_input('Username')
        new_password = st.text_input('Password', type='password')
        
        if st.button('Signup'):
            # create_usertable()
            add_userdata(new_user, new_password)
            st.success('Congratulation, you have successfully created an account')
            st.info('Go to the Login Menu to login')


    
  
elif choose == 'Data Exploration':
      st.title("Data Exploration for Loan Application")
      st.markdown('This section of the code is designated for performing various exploratory data analysis. You can upload your own file for the analysis. Also check below for more options of analysis')
      loan_file = st.file_uploader('Select Your Local Loan historical CSV (default provided)')
      if loan_file is not None:
            loan_df = pd.read_csv(loan_file)
            st.write(loan_dataset.head())
            st.write(loan_dataset.describe())
      else:
            loan_df= pd.read_csv('loan.csv')
            st.write(loan_dataset.head())
            st.write(loan_dataset.describe())
      
            st.subheader('Plotly Histogram Chart')
            selected_x_var = st.selectbox('What do want the x variable to be?', ['Gender','Self_Employed','Education'])
            fig = px.histogram(loan_df[selected_x_var])
            st.plotly_chart(fig)

            st.subheader('Plotly Bar Chart')
            selected_x_bar = st.selectbox('Select from the avaiable x variable: ', ['Married','Property_Area'])
            selected_y_bar = st.selectbox('Select from the avaiable y variable', ['Dependents','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term'])

            data = [go.Bar(
            x=loan_df[selected_x_bar], 
            y=loan_df[selected_y_bar]
            )]
            layout = go.Layout(
            title="Bar chart for: {},{}".format(selected_x_bar,selected_y_bar)
            )
            fig = go.Figure(data=data, layout=layout)
            st.plotly_chart(fig)


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

     
if __name__=='__main__': 
    main()