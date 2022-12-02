# import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk

# header information
st.title('Heart Disease Risk Calculator')

# read data file
disease_data = pd.read_csv('heart_data_binary.csv')


# create a function to find the percent of a specific population with heart disease
def disease_percent(data):
    total_count = data['HeartDisease'].count()
    num_disease = data[data['HeartDisease'] == 1]['HeartDisease'].count()
    disease_per = (num_disease / total_count) * 100
    return disease_per


# find the percent of each age group that has heart disease
diseasePer18 = disease_percent(disease_data[(disease_data['AgeCategory'] == '18-24')])
diseasePer25 = disease_percent(disease_data[(disease_data['AgeCategory'] == '25-29')])
diseasePer30 = disease_percent(disease_data[(disease_data['AgeCategory'] == '30-34')])
diseasePer35 = disease_percent(disease_data[(disease_data['AgeCategory'] == '35-39')])
diseasePer40 = disease_percent(disease_data[(disease_data['AgeCategory'] == '40-44')])
diseasePer45 = disease_percent(disease_data[(disease_data['AgeCategory'] == '45-49')])
diseasePer50 = disease_percent(disease_data[(disease_data['AgeCategory'] == '50-54')])
diseasePer55 = disease_percent(disease_data[(disease_data['AgeCategory'] == '55-59')])
diseasePer60 = disease_percent(disease_data[(disease_data['AgeCategory'] == '60-64')])
diseasePer65 = disease_percent(disease_data[(disease_data['AgeCategory'] == '65-69')])
diseasePer70 = disease_percent(disease_data[(disease_data['AgeCategory'] == '70-74')])
diseasePer75 = disease_percent(disease_data[(disease_data['AgeCategory'] == '75-79')])
diseasePer80 = disease_percent(disease_data[(disease_data['AgeCategory'] == '80 or older')])

# create a dictionary to map disease percentage by age category
riskByAge = {'18-24': diseasePer18, '25-29': diseasePer25, '30-34': diseasePer30, '35-39': diseasePer35,
             '40-44': diseasePer40, '45-49': diseasePer45, '50-54': diseasePer50, '55-59': diseasePer55,
             '60-64': diseasePer60, '65-69': diseasePer65, '70-74': diseasePer70, '75-79': diseasePer75,
             '80 or older': diseasePer80}

# create a line graph that shows the percentage of heart disease vs. age
st.write('Heart disease risk naturally rises with age:')
byAgeLine = plt.figure(figsize=(12, 4))
plt.xlabel('Age Category')
plt.ylabel('Risk Percentage')
plt.plot(riskByAge.keys(), riskByAge.values(), marker='o')
st.pyplot(byAgeLine)

# find the disease percentages for each sex category
diseasePerMale = disease_percent(disease_data[disease_data['Sex'] == 'Male'])
diseasePerFemale = disease_percent(disease_data[disease_data['Sex'] == 'Female'])

# create a bar graph that shows percentage of heart disease for males and females
st.write('Heart disease risk is also greater for biological men:')
sex_fig = plt.figure(figsize=(10, 4))
sns.barplot(x=['Female', 'Male'], y=[diseasePerFemale, diseasePerMale])
plt.ylabel('Risk Percentage')
st.pyplot(sex_fig)

# find the disease percentages for each BMI category
diseasePerHealthy = disease_percent(disease_data[disease_data['BMI'] < 25])
diseasePerOverweight = disease_percent(disease_data[(disease_data['BMI'] > 24) & (disease_data['BMI'] < 30)])
diseasePerObese = disease_percent(disease_data[disease_data['BMI'] > 29])

# find the disease percentages for each smoking category
diseasePerSmoker = disease_percent(disease_data[disease_data['Smoking'] == 'Yes'])
diseasePerNonSmoker = disease_percent(disease_data[disease_data['Smoking'] == 'No'])

# logistic regression variables
data_bmi = disease_data['BMI']
data_heartDisease = disease_data['HeartDisease']

# create graphs that show percentage of heart disease for BMI and Smoking
st.write('However, mutable lifestyle factors such as BMI and smoking can also increase heart disease risk:')
# logistic regression plot
bmi_fig = plt.figure()
sns.regplot(x=data_bmi, y=data_heartDisease, data=disease_data, logistic=True, ci=None)
plt.ylabel('Risk Percentage')
st.pyplot(bmi_fig)
# bar graph
smoking_fig = plt.figure()
sns.barplot(x=['Non-Smokers', 'Smokers'], y=[diseasePerNonSmoker, diseasePerSmoker])
plt.ylabel('Risk Percentage')
st.pyplot(smoking_fig)

# drop down menus for user to input stats
st.subheader("Calculate your heart disease risk:")
user_age = st.selectbox('Age Category', ('18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
                                         '60-64', '65-69', '70-74', '75-79', '80 or older'))
user_sex = st.selectbox('Sex', ('Male', 'Female'))
user_bmi = st.selectbox('BMI', (range(15, 101)))
user_smoking = st.selectbox('Do you smoke?', ('Yes', 'No'))

# create population matching the user
user_pop = disease_data[(disease_data['AgeCategory'] == user_age) & (disease_data['Sex'] == user_sex) & \
                        (disease_data['Smoking'] == user_smoking)]

user_pop_nonsmokers = disease_data[(disease_data['AgeCategory'] == user_age) & (disease_data['Sex'] == user_sex) & \
                                   (disease_data['Smoking'] == 'No')]

# create logistic regression predictions
user_x = user_pop['BMI'].values.reshape(-1, 1)
user_y = user_pop['HeartDisease'].values
user_logr = sk.linear_model.LogisticRegression()
user_logr.fit(user_x, user_y)


# create a function that calculates disease risk probability based on BMI
def disease_prob(logr, bmi):
    log_odds = logr.coef_ * bmi + logr.intercept_
    odds = np.exp(log_odds)
    prob = odds / (1 + odds)
    return float(prob[0] * 100)


# calculate disease risk probability
user_prob = disease_prob(user_logr, user_bmi)

# display risk information
user_input = st.button('Calculate Heart Disease Risk')
if user_input:
    st.write('Your risk of developing heart disease is', ("%.2f" % user_prob), "%.")

    if user_bmi < 25 and user_smoking == 'No':
        st.write('Please view the chart below and consult your doctor with any questions.')
        user_fig = plt.figure(figsize=(10, 4))
        sns.barplot(x=['Current Risk'], y=[user_prob])
        plt.ylabel('Risk Percentage')
        plt.ylim(0, (user_prob * 2))
        plt.title('Your Heart Disease Risk')
        st.pyplot(user_fig)

    if user_bmi > 24 and user_smoking == 'No':
        st.write('You can lower this risk by reducing your BMI.')
        user_prob_healthy = disease_prob(user_logr, 24)
        st.write('Reaching a healthy BMI of 24 reduces your heart disease risk to', ("%.2f" % user_prob_healthy), '%.')
        st.write('Please view the chart below and consult your doctor with any questions.')
        user_fig = plt.figure(figsize=(10, 4))
        sns.barplot(x=['Current BMI', 'BMI of 24'], y=[user_prob, user_prob_healthy])
        plt.ylabel('Risk Percentage')
        plt.title('Your Heart Disease Risk')
        st.pyplot(user_fig)

    if user_bmi < 25 and user_smoking == 'Yes':
        st.write('You can lower this risk by quitting smoking.')
        # create a non-smoking logistic regression model
        user_x2 = user_pop_nonsmokers['BMI'].values.reshape(-1, 1)
        user_y2 = user_pop_nonsmokers['HeartDisease'].values
        user_logr2 = linear_model.LogisticRegression()
        user_logr2.fit(user_x2, user_y2)
        user_prob2 = disease_prob(user_logr2, user_bmi)
        st.write('Quitting smoking reduces your heart disease risk to', ("%.2f" % user_prob2), '%.')
        st.write('Please view the chart below and consult your doctor with any questions.')
        user_fig = plt.figure(figsize=(10, 4))
        sns.barplot(x=['Smoking', 'Non-Smoking'], y=[user_prob, user_prob2])
        plt.ylabel('Risk Percentage')
        plt.title('Your Heart Disease Risk')
        st.pyplot(user_fig)

    if user_bmi > 24 and user_smoking == 'Yes':
        st.write('You can lower this risk by reducing your BMI.')
        user_prob_healthy = disease_prob(user_logr, 24)
        st.write('Reaching a healthy BMI of 24 reduces your heart disease risk to', ("%.2f" % user_prob_healthy), '%.')
        st.write('You can also lower this risk by quitting smoking.')
        user_x2 = user_pop_nonsmokers['BMI'].values.reshape(-1, 1)
        user_y2 = user_pop_nonsmokers['HeartDisease'].values
        user_logr2 = linear_model.LogisticRegression()
        user_logr2.fit(user_x2, user_y2)
        user_prob2 = disease_prob(user_logr2, user_bmi)
        st.write('Quitting smoking reduces your heart disease risk to', ("%.2f" % user_prob2), '%.')
        st.write('Please view the chart below and consult your doctor with any questions.')
        user_fig = plt.figure(figsize=(10, 4))
        sns.barplot(x=['Current Risk', 'BMI 24 Risk', 'Non-Smoking Risk'],
                    y=[user_prob, user_prob_healthy, user_prob2])
        plt.ylabel('Risk Percentage')
        plt.title('Your Heart Disease Risk')
        st.pyplot(user_fig)
