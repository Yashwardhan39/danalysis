#!/usr/bin/env python
# coding: utf-8

# In[4]:


# data_extraction.py

import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Retrieve credentials from .env
db_user = 'postgres'
db_password = 'y2aj3934'
db_host ='localhost'
db_port =  '5432'
db_name = 'JIVI_DB'

# Create connection string
connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'

# Create SQLAlchemy engine
engine = create_engine(connection_string)

# Test connection
try:
    with engine.connect() as connection:
        print("Connection to PostgreSQL database successful!")
except Exception as e:
    print(f"Error connecting to the database: {e}")


# In[5]:


import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Retrieve credentials from .env
db_user = 'postgres'
db_password = 'y2aj3934'
db_host ='localhost'
db_port =  '5432'
db_name = 'JIVI_DB'

# Create connection string
connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'

# Create SQLAlchemy engine
engine = create_engine(connection_string)

# Test connection
try:
    with engine.connect() as connection:
        print("Connection to PostgreSQL database successful!")
except Exception as e:
    print(f"Error connecting to the database: {e}")


# In[1]:


pip install pandas sqlalchemy psycopg2-binary python-dotenv seaborn


# In[12]:


import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# credentials from .env
db_user = 'postgres'
db_password = 'y2aj3934'
db_host ='localhost'
db_port =  '5432'
db_name = 'JIVI_DB'

# Create connection string
connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'

# Create SQLAlchemy engine
engine = create_engine(connection_string)

# Test connection
try:
    with engine.connect() as connection:
        print("Connection to PostgreSQL database successful!")
except Exception as e:
    print(f"Error connecting to the database: {e}")
# Define SQL queries for each table
queries = {
    'Users': 'SELECT * FROM "users";',
    'Symptoms': 'SELECT * FROM "symptoms";',
    'UserSymptoms': 'SELECT * FROM "usersymptoms";',
    'Diagnoses': 'SELECT * FROM "diagnoses";',
    'UserDiagnoses': 'SELECT * FROM "userdiagnoses";',
    'Doctors': 'SELECT * FROM "doctors";',
    'UserDoctorRecommendations': 'SELECT * FROM "userdoctorrecommendations";',
    'Medications': 'SELECT * FROM "medications";',
    'UserMedications': 'SELECT * FROM "usermedications";',
    'Remedies': 'SELECT * FROM "remedies";',
    'UserRemedies': 'SELECT * FROM "userremedies";',
    'DietCharts': 'SELECT * FROM "dietcharts";',
    'HealthcareCharts': 'SELECT * FROM "healthcarecharts";'
}

# Extract data into DataFrames
data_frames = {}
for table, query in queries.items():
    try:
        df = pd.read_sql_query(query,engine)
        data_frames[table] = df
        print(f"Extracted {len(df)} records from {table} table.")
    except Exception as e:
        print(f"Error extracting data from {table}: {e}")

# Example: Display first few rows of Users table
if 'Users' in data_frames:
    print("\nUsers DataFrame:")
    print(data_frames['Users'].head())


# In[13]:


pip install sqlalchemy pandas psycopg2-binary


# In[1]:


import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# credentials from .env
db_user = 'postgres'
db_password = 'y2aj3934'
db_host ='localhost'
db_port =  '5432'
db_name = 'JIVI_DB'

# Create connection string
connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'

# Create SQLAlchemy engine
engine = create_engine(connection_string)

# Test connection
try:
    with engine.connect() as connection:
        print("Connection to PostgreSQL database successful!")
except Exception as e:
    print(f"Error connecting to the database: {e}")
# Define SQL queries for each table
queries = {
    'Users': 'SELECT * FROM "users";',
    'Symptoms': 'SELECT * FROM "symptoms";',
    'UserSymptoms': 'SELECT * FROM "usersymptoms";',
    'Diagnoses': 'SELECT * FROM "diagnoses";',
    'UserDiagnoses': 'SELECT * FROM "userdiagnoses";',
    'Doctors': 'SELECT * FROM "doctors";',
    'UserDoctorRecommendations': 'SELECT * FROM "userdoctorrecommendations";',
    'Medications': 'SELECT * FROM "medications";',
    'UserMedications': 'SELECT * FROM "usermedications";',
    'Remedies': 'SELECT * FROM "remedies";',
    'UserRemedies': 'SELECT * FROM "userremedies";',
    'DietCharts': 'SELECT * FROM "dietcharts";',
    'HealthcareCharts': 'SELECT * FROM "healthcarecharts";'
}

# Extract data into DataFrames
data_frames = {}
for table, query in queries.items():
    try:
        df = pd.read_sql_query(query,engine)
        data_frames[table] = df
        print(f"Extracted {len(df)} records from {table} table.")
    except Exception as e:
        print(f"Error extracting data from {table}: {e}")

# Example: Display first few rows of Users table
if 'Users' in data_frames:
    print("\nUsers DataFrame:")
    print(data_frames['Users'].head())


# In[3]:


pip show pandas sqlalchemy psycopg2-binary python-dotenv


# In[2]:


# data_extraction.py

import pandas as pd
from sqlalchemy import create_engine

# Retrieve credentials 
db_user = 'postgres'
db_password = 'y2aj3934'
db_host ='localhost'
db_port ='5432'
db_name = 'JIVI_DB'

# Create connection string
connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'

# Create SQLAlchemy engine
engine = create_engine(connection_string)

# Test connection
try:
    with engine.connect() as connection:
        print("Connection to PostgreSQL database successful!")
except Exception as e:
    print(f"Error connecting to the database: {e}")
    exit()

# Define SQL queries for each table
queries = {
    'Users': 'SELECT * FROM "Users";',
    'Symptoms': 'SELECT * FROM "Symptoms";',
    'UserSymptoms': 'SELECT * FROM "UserSymptoms";',
    'Diagnoses': 'SELECT * FROM "Diagnoses";',
    'UserDiagnoses': 'SELECT * FROM "UserDiagnoses";',
    'Doctors': 'SELECT * FROM "Doctors";',
    'UserDoctorRecommendations': 'SELECT * FROM "UserDoctorRecommendations";',
    'Medications': 'SELECT * FROM "Medications";',
    'UserMedications': 'SELECT * FROM "UserMedications";',
    'Remedies': 'SELECT * FROM "Remedies";',
    'UserRemedies': 'SELECT * FROM "UserRemedies";',
    'DietCharts': 'SELECT * FROM "DietCharts";',
    'HealthcareCharts': 'SELECT * FROM "HealthcareCharts";'
}

# Extract data into DataFrames
data_frames = {}
for table, query in queries.items():
    try:
        df = pd.read_sql(query, engine)
        data_frames[table] = df
        print(f"Extracted {len(df)} records from {table} table.")
    except Exception as e:
        print(f"Error extracting data from {table}: {e}")

# Example: Display first few rows of Users table
if 'Users' in data_frames:
    print("\nUsers DataFrame:")
    print(data_frames['Users'].head())


# In[2]:


#Installing all required libraries
import pandas as pd
import pandas.io.sql as sqlio
import psycopg2 as ps
#Connecting with Postgresql 
conn2=ps.connect(dbname="JIVI_DB",
                user='postgres',
                password='y2aj3934',
                host='localhost',
                port='5432')
try:
        print("Connection to PostgreSQL database successful!")
except Exception as e:
    print(f"Error connecting to the database: {e}")
    exit()
    
queries = {
    'Users': 'SELECT * FROM "users";',
    'Symptoms': 'SELECT * FROM "symptoms";',
    'UserSymptoms': 'SELECT * FROM "usersymptoms";',
    'Diagnoses': 'SELECT * FROM "diagnoses";',
    'UserDiagnoses': 'SELECT * FROM "userdiagnoses";',
    'Doctors': 'SELECT * FROM "doctors";',
    'UserDoctorRecommendations': 'SELECT * FROM "userdoctorrecommendations";',
    'Medications': 'SELECT * FROM "medications";',
    'UserMedications': 'SELECT * FROM "usermedications";',
    'Remedies': 'SELECT * FROM "remedies";',
    'UserRemedies': 'SELECT * FROM "userremedies";',
    'DietCharts': 'SELECT * FROM "dietcharts";',
    'HealthcareCharts': 'SELECT * FROM "healthcarecharts";'
}

# Extract data into DataFrames
data_frames = {}
for table, query in queries.items():
    try:
        df = sqlio.read_sql_query(query, conn2)
        data_frames[table] = df
        print(f"Extracted {len(df)} records from {table} table.")
    except Exception as e:
        print(f"Error extracting data from {table}: {e}")
print()
print()

#Displaying names of all the records 
print("Table Names found in our backend schema")
print()
for i in queries:
    print(i)
print()
# Example: Display first few rows of any table
a=input("Enter the name of the table to view records:-")
if a in data_frames:
    print("\n",a, "DataFrame:")
    print(data_frames[a].head())


# In[3]:


# Check for missing values in each DataFrame
for table, df in data_frames.items():
    print(f"\nMissing values in {table}:")
    print(df.isnull().sum())


# In[4]:


data_frames['Users'].dropna(inplace=True)


# In[7]:


# Fill missing numerical values with median
data_frames['UserSymptoms']['severity'].fillna(data_frames['UserSymptoms']['severity'].median(), inplace=True)

# Fill missing categorical values with mode
data_frames['Users']['gender'].fillna(data_frames['Users']['gender'].mode()[0], inplace=True)


# In[8]:


# Convert 'RegistrationDate' to datetime
data_frames['Users']['registrationdate'] = pd.to_datetime(data_frames['Users']['registrationdate'], errors='coerce')

# Convert numerical columns to appropriate types
data_frames['UserSymptoms']['severity'] = pd.to_numeric(data_frames['UserSymptoms']['severity'], errors='coerce')


# In[9]:


# Remove duplicate rows in the Users table
data_frames['Users'].drop_duplicates(inplace=True)


# In[10]:


# Example: Extract year from RegistrationDate
data_frames['Users']['registrationyear'] = data_frames['Users']['registrationdate'].dt.year

# Example: Calculate duration in days if Duration is in a string format like '3 days'
data_frames['UserSymptoms']['durationdays'] = data_frames['UserSymptoms']['duration'].str.extract('(\d+)').astype(float)


# In[11]:


# Standardize Gender entries to capitalize the first letter
data_frames['Users']['gender'] = data_frames['Users']['gender'].str.capitalize()

# Example: Replace abbreviations
data_frames['Users']['gender'].replace({'M': 'Male', 'F': 'Female'}, inplace=True)


# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt

# Visualize severity distribution
sns.boxplot(x=data_frames['UserSymptoms']['severity'])
plt.title('Severity Distribution')
plt.show()

# Remove outliers beyond 1.5*IQR
Q1 = data_frames['UserSymptoms']['severity'].quantile(0.25)
Q3 = data_frames['UserSymptoms']['severity'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the DataFrame
data_frames['UserSymptoms'] = data_frames['UserSymptoms'][
    (data_frames['UserSymptoms']['severity'] >= lower_bound) &
    (data_frames['UserSymptoms']['severity'] <= upper_bound)
]


# In[14]:


# Save the cleaned Users DataFrame to a CSV file
data_frames['Users'].to_csv('cleaned_users.csv', index=False)

# Similarly, save other tables as needed


# In[20]:



# Function to clean a specific table
def clean_table(table_name, df):
    print(f"\nCleaning table: {table_name}")
    
    if table_name == 'Users':
        # Handle missing values
        df['age'].fillna(df['age'].median(), inplace=True)
        df['gender'].fillna(df['gender'].mode()[0], inplace=True)
        
        # Convert RegistrationDate to datetime
        df['registrationdate'] = pd.to_datetime(df['registrationdate'], errors='coerce')
        
        # Remove duplicates based on UserID
        df.drop_duplicates(subset='userid', inplace=True)
        
    elif table_name == 'Symptoms':
        # Remove duplicates based on SymptomID
        df.drop_duplicates(subset='symptomid', inplace=True)
        
    elif table_name == 'UserSymptoms':
        # Convert Severity to numeric
        df['severity'] = pd.to_numeric(df['severity'], errors='coerce')
        df['severity'].fillna(df['severity'].median(), inplace=True)
        
        # Extract DurationDays from Duration string
        df['durationdays'] = df['duration'].str.extract('(\d+)').astype(float)
        df['durationdays'].fillna(df['durationdays'].median(), inplace=True)
        
        # Remove duplicates based on UserID and SymptomID
        df.drop_duplicates(subset=['userid', 'symptomid'], inplace=True)
        
    elif table_name == 'Diagnoses':
        # Remove duplicates based on DiagnosisID
        df.drop_duplicates(subset='diagnosisid', inplace=True)
        
    elif table_name == 'UserDiagnoses':
        # Convert ConfidenceScore to numeric
        df['confidencescore'] = pd.to_numeric(df['confidencescore'], errors='coerce')
        df['confidencescore'].fillna(df['confidencescore'].median(), inplace=True)
        
        # Remove duplicates based on UserID and DiagnosisID
        df.drop_duplicates(subset=['userid', 'diagnosisid'], inplace=True)
        
    elif table_name == 'Doctors':
        # Handle missing values
        df['specialization'].fillna('General', inplace=True)
        
        # Remove duplicates based on DoctorID
        df.drop_duplicates(subset='doctorid', inplace=True)
        
    elif table_name == 'UserDoctorRecommendations':
        # Convert RecommendationDate to datetime
        df['recommendationdate'] = pd.to_datetime(df['recommendationdate'], errors='coerce')
        
        # Remove duplicates based on UserID and DoctorID
        df.drop_duplicates(subset=['userid', 'doctorid'], inplace=True)
        
    elif table_name == 'Medications':
        # Remove duplicates based on MedicationID
        df.drop_duplicates(subset='medicationid', inplace=True)
        
    elif table_name == 'UserMedications':
        # Convert StartDate and EndDate to datetime
        df['startdate'] = pd.to_datetime(df['startdate'], errors='coerce')
        df['enddate'] = pd.to_datetime(df['enddate'], errors='coerce')
        
        # Remove duplicates based on UserID and MedicationID
        df.drop_duplicates(subset=['userid', 'medicationid'], inplace=True)
        
    elif table_name == 'Remedies':
        # Remove duplicates based on RemedyID
        df.drop_duplicates(subset='remedyid', inplace=True)
        
    elif table_name == 'UserRemedies':
        # Convert UsageFrequency to numeric
        df['usagefrequency'] = pd.to_numeric(df['usagefrequency'], errors='coerce')
        df['usagefrequency'].fillna(df['usagefrequency'].median(), inplace=True)
        
        # Remove duplicates based on UserID and RemedyID
        df.drop_duplicates(subset=['userid', 'remedyid'], inplace=True)
        
    elif table_name == 'DietCharts':
        # Remove duplicates based on ChartID
        df.drop_duplicates(subset='dietchartid', inplace=True)
        
    elif table_name == 'HealthcareCharts':
        # Convert Value to numeric and Date to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Remove duplicates based on ChartID
        df.drop_duplicates(subset='chartid', inplace=True)
        
    else:
        print(f"No cleaning rules defined for table: {table_name}")
        return df
    # For simplicity, we'll proceed to save the cleaned DataFrame
    
    # Save the cleaned DataFrame to a CSV file
    clean_file_name = f'cleaned_{table_name}.csv'
    df.to_csv(clean_file_name, index=False)
    print(f"Cleaned data saved to {clean_file_name}")
    
    return df

# Infinite loop to clean tables based on user input
while True:
    user_input = input("\nEnter the table name you want to clean (or type 'jivi' to exit): ")
    
    # Check if the user wants to exit
    if user_input.strip().lower() == 'jivi':
        print("Exiting the cleaning process. All desired tables have been cleaned.")
        break
    
    # Check if the entered table name exists in data_frames
    if user_input in data_frames:
        # Perform cleaning
        data_frames[user_input] = clean_table(user_input, data_frames[user_input])
    else:
        print(f"Table '{user_input}' does not exist. Please enter a valid table name or type 'Jivi' to exit.")

#We can also implement an if condition for couloumn name to ensure a beautyful cleaning process for example
# elif table_name == 'Remedies':
        # Remove duplicates based on RemedyID
#        if 'RemedyID' in df.columns:
#            duplicates = df.duplicated(subset='RemedyID').sum()
#            df.drop_duplicates(subset='RemedyID', inplace=True)
#            print(f" - Removed {duplicates} duplicate records based on 'RemedyID'.")


# In[24]:


pip install matplotlib seaborn


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For better visualization aesthetics
sns.set(style="whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


# Example: Loading the cleaned Users table
users_df = pd.read_csv('cleaned_Users.csv')
symptoms_df = pd.read_csv('cleaned_Symptoms.csv')
user_symptoms_df = pd.read_csv('cleaned_UserSymptoms.csv')
diagnoses_df=pd.read_csv('cleaned_Diagnoses.csv')
user_diagnoses_df=pd.read_csv('cleaned_UserDiagnoses.csv')
doctors_df=pd.read_csv('cleaned_Doctors.csv')
user_doctor_recommendations_df=pd.read_csv('cleaned_UserDoctorRecommendations.csv')
medications_df=pd.read_csv('cleaned_Medications.csv')
user_medications_df=pd.read_csv('cleaned_UserMedications.csv')
remedies_df=pd.read_csv('cleaned_Remedies.csv')
user_remedies_df=pd.read_csv('cleaned_UserRemedies.csv')
diet_charts_df=pd.read_csv('cleaned_DietCharts.csv')
health_care_charts_df=pd.read_csv('cleaned_HealthcareCharts.csv')


# In[9]:


# Display the first few rows
print("Users DataFrame:")
print(users_df.head())

print("\nSymptoms DataFrame:")
print(symptoms_df.head())

print("\nUserSymptoms DataFrame:")
print(user_symptoms_df.head())

print("\nDiagnoses DataFrame:")
print(diagnoses_df.head())

print("\nUserDiagnoses DataFrame:")
print(user_diagnoses_df.head())

print("\nDoctors DataFrame:")
print(doctors_df.head())

print("\nUserDoctorRecommendations DataFrame:")
print(user_doctor_recommendations_df.head())

print("\nMedications DataFrame:")
print(medications_df.head())

print("\nUserMedications DataFrame:")
print(user_medications_df.head())

print("\nRemedies DataFrame:")
print(remedies_df.head())

print("\nUserRemedies DataFrame:")
print(user_remedies_df.head())

print("\nDietCharts DataFrame:")
print(diet_charts_df.head())

print("\nHealthcareCharts DataFrame:")
print(health_care_charts_df.head())




# Check data types and non-null counts
print("\nUsers DataFrame Info:")
print(users_df.info())

print("\nSymptoms DataFrame Info:")
print(symptoms_df.info())

print("\nUserSymptoms DataFrame Info:")
print(user_symptoms_df.info())

print("\nDiagnoses DataFrame Info:")
print(diagnoses_df.info())

print("\nUserDiagnoses DataFrame Info:")
print(user_diagnoses_df.info())

print("\nDoctors DataFrame Info:")
print(doctors_df.info())

print("\nUserDoctorRecommendations DataFrame Info:")
print(user_doctor_recommendations_df.info())

print("\nMedications DataFrame Info:")
print(medications_df.info())

print("\nUserMedications DataFrame Info:")
print(user_medications_df.info())

print("\nRemedies DataFrame Info:")
print(remedies_df.info())

print("\nUserRemedies DataFrame Info:")
print(user_remedies_df.info())

print("\nDietCharts DataFrame Info:")
print(diet_charts_df.info())

print("\nHealthcareCharts DataFrame Info:")
print(health_care_charts_df.info())



# In[10]:


#Summary Statistics for all the tables
print("Users Summary Statistics:")
print(users_df.describe())

print("\nUserSymptoms Summary Statistics:")
print(user_symptoms_df.describe())

print("\nUserSymptoms Statistics:")
print(user_symptoms_df.describe())

print("\nDiagnoses Statistics:")
print(diagnoses_df.describe())

print("\nUserDiagnoses Statistics:")
print(user_diagnoses_df.describe())

print("\nDoctors Statistics:")
print(doctors_df.describe())

print("\nUserDoctorRecommendations Statistics:")
print(user_doctor_recommendations_df.describe())

print("\nMedications Statistics:")
print(medications_df.describe())

print("\nUserMedications Statistics:")
print(user_medications_df.describe())

print("\nRemedies Statistics:")
print(remedies_df.describe())

print("\nUserRemedies Statistics:")
print(user_remedies_df.describe())

print("\nDietCharts Statistics:")
print(diet_charts_df.describe())

print("\nHealthcareCharts Statistics:")
print(health_care_charts_df.describe())


# In[7]:


#Visualisation for Tables
#Histogram for age in Users Table
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For better aesthetics
sns.set(style="whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')

# Load the CSV file into a DataFrame
df = pd.read_csv('cleaned_users.csv')

# Check if the 'Age' column exists and has valid data
if 'age' in df.columns:
    # Drop rows where 'Age' is missing or not a number (optional)
    df = df.dropna(subset=['age'])
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df.dropna(subset=['age'])

    # Create the histogram for the 'Age' column
    plt.figure(figsize=(10, 6))
    plt.hist(df['age'], bins=10, edgecolor='black')

    # Add labels and title
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()
else:
    print("The 'Age' column was not found in the CSV file.")

plt.figure(figsize=(8,6))
sns.countplot(x='gender', data=users_df, palette='pastel')
plt.title('Gender Distribution of Users')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# In[10]:



users_df['registrationdate'] = pd.to_datetime(users_df['registrationdate'])

# Group by month/year
registrations = users_df.set_index('registrationdate').resample('M').size()

plt.figure(figsize=(12,6))
registrations.plot(kind='line', marker='o', color='green')
plt.title('User Registrations Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Registrations')
plt.show()


# In[14]:


# Create Age Groups if not already done
users_df['AgeGroup'] = pd.cut(users_df['age'], bins=[0, 18, 35, 50, 65, 100], labels=['<18', '18-35', '35-50', '50-65', '65+'], right=False)

plt.figure(figsize=(10,6))
sns.countplot(x='AgeGroup', data=users_df, palette='Set2')
plt.title('Age Group Distribution of Users')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()


# In[17]:


plt.figure(figsize=(12,8))
top_symptoms = symptoms_df['symptomdescription'].value_counts().head(10)
sns.barplot(x=top_symptoms.values, y=top_symptoms.index, palette='viridis')
plt.title('Top 10 Most Common Symptoms')
plt.xlabel('Number of Occurrences')
plt.ylabel('Symptom')
plt.show()


# In[24]:


plt.figure(figsize=(10,6))
sns.boxplot(x='symptomid', y='severity', data=user_symptoms_df, palette='Set3')
plt.title('Severity Distribution Across Symptoms')
plt.xlabel('Symptom')
plt.ylabel('Severity')
plt.xticks(rotation=45)
plt.show()


# In[27]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('cleaned_UserSymptoms.csv')

# Check if the 'durationdays' column exists and has valid data
if 'durationdays' in df.columns:
    # Drop rows where 'durationdays' is missing or not a number (optional)
    df = df.dropna(subset=['durationdays'])
    df['durationdays'] = pd.to_numeric(df['durationdays'], errors='coerce')
    df = df.dropna(subset=['durationdays'])

    # Create the histogram for the 'durationdays' column
    plt.figure(figsize=(10, 6))
    plt.hist(df['durationdays'], bins=10, edgecolor='black')

    # Add labels and title
    plt.title('Distribution of Duration Days')
    plt.xlabel('Duration (Days)')
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()
else:
    print("The 'durationdays' column was not found in the CSV file.")


# In[38]:


''''# Count symptoms per user
symptoms_per_user = user_symptoms_df['userid'].value_counts()

plt.figure(figsize=(10,6))
plt.hist(symptoms_per_user, bins=20, kde=False, color='teal')
plt.title('Number of Symptoms Reported per User')
plt.xlabel('Number of Symptoms')
plt.ylabel('Number of Users')
plt.show()


# Drop rows where 'durationdays' is missing or not a number (optional)
user_symptoms_df = user_symptoms_df.dropna(subset=['userid'])
user_symptoms_df['userid'] = pd.to_numeric(user_symptoms_df['userid'], errors='coerce')
user_symptoms_df = user_symptoms_df.dropna(subset=['userid'])

# Create the histogram for the 'durationdays' column
plt.figure(figsize=(10, 6))
plt.hist(user_symptoms_df['userid'], bins=10, edgecolor='black')

# Add labels and title
plt.title('Number of Symptoms Reported per user')
plt.xlabel('Number of Symptoms')
plt.ylabel('Number of Users')

# Show the plot
plt.show()


# In[39]:


plt.figure(figsize=(10,6))
sns.scatterplot(x='severity', y='durationdays', data=user_symptoms_df, alpha=0.6)
sns.regplot(x='severity', y='durationdays', data=user_symptoms_df, scatter=False, color='red')
plt.title('Correlation Between Symptom Severity and Duration')
plt.xlabel('Severity')
plt.ylabel('Duration Days')
plt.show()


# In[45]:


user_symptoms_df['datereported'] = pd.to_datetime(user_symptoms_df['datereported'], errors='coerce')
symptoms_over_time = user_symptoms_df.set_index('datereported').resample('M').size()

plt.figure(figsize=(12,6))
sns.lineplot(x=symptoms_over_time.index, y=symptoms_over_time.values, marker='o', color='purple')
plt.title('Number of Symptoms Reported Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Symptoms')
plt.show()


# In[47]:


# Convert datereported to datetime format (if not already done)
user_symptoms_df['datereported'] = pd.to_datetime(user_symptoms_df['datereported'])

# Group by datereported and count the number of symptoms reported each day
symptoms_count = user_symptoms_df.groupby('datereported').size().reset_index(name='num_symptoms')

# Sort by date (optional, if not already sorted)
symptoms_count = symptoms_count.sort_values(by='datereported')

# Plotting the line chart
plt.figure(figsize=(10, 6))
plt.plot(symptoms_count['datereported'], symptoms_count['symptomid'], marker='o', linestyle='-')
plt.title('Number of Symptoms Reported Over Time')
plt.xlabel('Date Reported')
plt.ylabel('Number of Symptoms')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[53]:


#Ensure the 'Date' column is in datetime format
user_symptoms_df['datereported'] = pd.to_datetime(user_symptoms_df['datereported'], errors='coerce')
import pandas as pd
import seaborn as sns
# Correctly set the option to treat infinities as NaN
pd.set_option('mode.use_inf_as_na', True)
# Set 'Date' as the DataFrame index and resample the data monthly
symptoms_over_time = user_symptoms_df.set_index('datereported').resample('M').size()

# Create the plot
plt.figure(figsize=(12,6))
sns.lineplot(x=symptoms_over_time.index, y=symptoms_over_time.values, marker='o', color='purple')
plt.title('Number of Symptoms Reported Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Symptoms')
plt.show()


# In[51]:


import pandas as pd

print(pd.__version__)


# In[52]:


import pandas as pd

# Check if the option exists
available_options = pd.describe_option('mode')
print(available_options)


# In[56]:


plt.figure(figsize=(12,8))
top_diagnoses = diagnoses_df['diagnosisname'].value_counts().head(10)
sns.barplot(x=top_diagnoses.values, y=top_diagnoses.index, palette='magma')
plt.title('Top 10 Most Common Diagnoses')
plt.xlabel('Number of Occurrences')
plt.ylabel('Diagnosis')
plt.show()


# In[62]:


user_diagnoses_df = pd.merge(user_diagnoses_df, users_df, on='userid')

diagnosis_gender = user_diagnoses_df.groupby(['diagnosisid', 'userid']).size().unstack(fill_value=0)

diagnosis_gender.plot(kind='bar', stacked=True, figsize=(12,8), colormap='Accent')
plt.title('Diagnosis Distribution by Gender')
plt.xlabel('Diagnosis')
plt.ylabel('Number of Occurrences')
plt.legend(title='Gender')
plt.show()


# In[63]:


plt.figure(figsize=(10,6))
sns.histplot(user_diagnoses_df['confidencescore'], bins=20, kde=True, color='orange')
plt.title('Distribution of Confidence Scores in Diagnoses')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.show()


# In[65]:


#Plotting the histogram for 'confidence_score'
user_diagnoses_df['confidencescore'].hist(bins=10, edgecolor='black')

# Adding title and labels
plt.title('Histogram of Confidence Scores')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')

# Display the plot
plt.show()


# In[66]:



plt.figure(figsize=(10,8))
specialty_counts = doctors_df['specialization'].value_counts()
sns.barplot(x=specialty_counts.values, y=specialty_counts.index, palette='viridis')
plt.title('Distribution of Doctor Specialties')
plt.xlabel('Number of Doctors')
plt.ylabel('Specialty')
plt.show()


# In[67]:


plt.figure(figsize=(12,8))
sns.countplot(y='specialization', data=doctors_df, order=doctors_df['specialization'].value_counts().index, palette='coolwarm')
plt.title('Number of Doctors per Specialty')
plt.xlabel('Count')
plt.ylabel('Specialty')
plt.show()


# In[68]:



user_doctor_recommendations_df['recommendationdate'] = pd.to_datetime(user_doctor_recommendations_df['recommendationdate'], errors='coerce')

recommendations_over_time = user_doctor_recommendations_df.set_index('recommendationdate').resample('M').size()

plt.figure(figsize=(12,6))
sns.lineplot(x=recommendations_over_time.index, y=recommendations_over_time.values, marker='o', color='brown')
plt.title('Doctor Recommendations Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Recommendations')
plt.show()


# In[69]:


plt.figure(figsize=(12,8))
top_medications = medications_df['medicationname'].value_counts().head(10)
sns.barplot(x=top_medications.values, y=top_medications.index, palette='plasma')
plt.title('Top 10 Most Prescribed Medications')
plt.xlabel('Number of Prescriptions')
plt.ylabel('Medication')
plt.show()


# In[70]:





# In[72]:


medications_per_user = user_medications_df['userid'].value_counts()

plt.figure(figsize=(10,6))
sns.histplot(medications_per_user, bins=20, kde=False, color='lightgreen')
plt.title('Number of Medications per User')
plt.xlabel('Number of Medications')
plt.ylabel('Number of Users')
plt.show()


# In[73]:


# Assuming 'StartDate' and 'EndDate' exist
user_medications_df['startdate'] = pd.to_datetime(user_medications_df['startdate'], errors='coerce')
user_medications_df['enddate'] = pd.to_datetime(user_medications_df['enddate'], errors='coerce')

# Calculate duration
user_medications_df['Duration'] = (user_medications_df['enddate'] - user_medications_df['startdate']).dt.days

# Plot average duration over time
average_duration = user_medications_df.set_index('startdate').resample('M')['Duration'].mean()

plt.figure(figsize=(12,6))
sns.lineplot(x=average_duration.index, y=average_duration.values, marker='o', color='blue')
plt.title('Average Medication Adherence Duration Over Time')
plt.xlabel('Date')
plt.ylabel('Average Duration (Days)')
plt.show()


# In[74]:


# Merge with Users to get age
user_medications_df = pd.merge(user_medications_df, users_df[['userid', 'age']], on='userid', how='left')

plt.figure(figsize=(12,8))
sns.boxplot(x='age', y='dosage', data=user_medications_df, palette='coolwarm')
plt.title('Medication Dosage by Age Group')
plt.xlabel('Age ')
plt.ylabel('Dosage')
plt.show()


# In[75]:


plt.figure(figsize=(12,8))
top_remedies = remedies_df['remedyname'].value_counts().head(10)
sns.barplot(x=top_remedies.values, y=top_remedies.index, palette='cividis')
plt.title('Top 10 Most Common Remedies')
plt.xlabel('Number of Uses')
plt.ylabel('Remedy')
plt.show()


# In[76]:


remedies_per_user = user_remedies_df['userid'].value_counts()

plt.figure(figsize=(10,6))
sns.histplot(remedies_per_user, bins=20, kde=False, color='gold')
plt.title('Number of Remedies Used per User')
plt.xlabel('Number of Remedies')
plt.ylabel('Number of Users')
plt.show()


# In[77]:


plt.figure(figsize=(12,8))
usage_frequency = user_remedies_df['usagefrequency'].value_counts().head(10)
sns.barplot(x=usage_frequency.values, y=usage_frequency.index, palette='inferno')
plt.title('Top 10 Most Frequently Used Remedies')
plt.xlabel('Usage Frequency')
plt.ylabel('Remedy')
plt.show()


# In[79]:


diet_charts_df['creationdate'] = pd.to_datetime(diet_charts_df['creationdate'], errors='coerce')
adherence_over_time = diet_charts_df.set_index('creationdate').resample('M').size()
plt.figure(figsize=(12,6))
sns.lineplot(x=adherence_over_time.index, y=adherence_over_time.values, marker='o', color='cyan')
plt.title('Diet Adherence Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Adherences')
plt.show()


# In[ ]:


while True:
    print()
    print("Enter the table name to view suggested Visuals")
    print("Enter Jivi to exit")
    print()
    a=input("Enter the Table Name:-")
    if a=="Users" or "users":
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        # For better aesthetics
        sns.set(style="whitegrid")
        get_ipython().run_line_magic('matplotlib', 'inline')
        print()
        print("Visualization: Histogram with KDE (Kernel Density Estimate)")
        print()

        # Load the CSV file into a DataFrame
        df = pd.read_csv('cleaned_users.csv')

        # Check if the 'Age' column exists and has valid data
        if 'age' in df.columns:
            # Drop rows where 'Age' is missing or not a number (optional)
            df = df.dropna(subset=['age'])
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            df = df.dropna(subset=['age'])

            # Create the histogram for the 'Age' column
            plt.figure(figsize=(10, 6))
            plt.hist(df['age'], bins=10, edgecolor='black')

            # Add labels and title
            plt.title('Age Distribution')
            plt.xlabel('Age')
            plt.ylabel('Frequency')

            # Show the plot
            plt.show()
        else:
            print("The 'Age' column was not found in the CSV file.")
        print()
        print("Visualization: Count Plot")
        print()
        plt.figure(figsize=(8,6))
        sns.countplot(x='gender', data=users_df, palette='pastel')
        plt.title('Gender Distribution of Users')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.show()
        print()
        print("Visualization: Line Plot showing number of registrations per month/year")
        print()
        users_df['registrationdate'] = pd.to_datetime(users_df['registrationdate'])
        # Group by month/year
        registrations = users_df.set_index('registrationdate').resample('M').size()

        plt.figure(figsize=(12,6))
        registrations.plot(kind='line', marker='o', color='green')
        plt.title('User Registrations Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Registrations')
        plt.show()
        print()
        print("Visualisation for different age Groups:-")
        print()
        # Create Age Groups if not already done
        users_df['AgeGroup'] = pd.cut(users_df['age'], bins=[0, 18, 35, 50, 65, 100], labels=['<18', '18-35', '35-50', '50-65', '65+'], right=False)
        plt.figure(figsize=(10,6))
        sns.countplot(x='AgeGroup', data=users_df, palette='Set2')
        plt.title('Age Group Distribution of Users')
        plt.xlabel('Age Group')
        plt.ylabel('Count')
        plt.show()
    
    elif a=="Symptoms" or "symptoms":
        print()
        print("Visualisation for top 10 SYMPTOMS")
        print()
        plt.figure(figsize=(12,8))
        top_symptoms = symptoms_df['symptomdescription'].value_counts().head(10)
        sns.barplot(x=top_symptoms.values, y=top_symptoms.index, palette='viridis')
        plt.title('Top 10 Most Common Symptoms')
        plt.xlabel('Number of Occurrences')
        plt.ylabel('Symptom')
        plt.show()
        print()
        print("Visualisation of Symptoms severity based on symptomid")
        print()
        plt.figure(figsize=(10,6))
        sns.boxplot(x='symptomid', y='severity', data=user_symptoms_df, palette='Set3')
        plt.title('Severity Distribution Across Symptoms')
        plt.xlabel('Symptom')
        plt.ylabel('Severity')
        plt.xticks(rotation=45)
        plt.show()
        print()
        print("Visualisation of The number of days a particular duration lasts")
        print()
        # Load the CSV file into a DataFrame
        df = pd.read_csv('cleaned_UserSymptoms.csv')

        # Check if the 'durationdays' column exists and has valid data
        if 'durationdays' in df.columns:
            # Drop rows where 'durationdays' is missing or not a number (optional)
            df = df.dropna(subset=['durationdays'])
            df['durationdays'] = pd.to_numeric(df['durationdays'], errors='coerce')
            df = df.dropna(subset=['durationdays'])

            # Create the histogram for the 'durationdays' column
            plt.figure(figsize=(10, 6))
            plt.hist(df['durationdays'], bins=10, edgecolor='black')

            # Add labels and title
            plt.title('Distribution of Duration Days')
            plt.xlabel('Duration (Days)')
            plt.ylabel('Frequency')

            # Show the plot
            plt.show()
        else:
            print("The 'durationdays' column was not found in the CSV file.")
    elif a=="UserSymptoms" or "usersymptoms":
        print()
        print("Graph for Severity of disease VS its Duration")
        print()
        plt.figure(figsize=(10,6))
        sns.scatterplot(x='severity', y='durationdays', data=user_symptoms_df, alpha=0.6)
        sns.regplot(x='severity', y='durationdays', data=user_symptoms_df, scatter=False, color='red')
        plt.title('Correlation Between Symptom Severity and Duration')
        plt.xlabel('Severity')
        plt.ylabel('Duration Days')
        plt.show()
        print()
        print(" 2 graphn banane h yha abhiii")
        print()
    elif a=="Diagnoses" or "diagnoses":
        print()
        print("Bar Chart for Most Common Diagnosis")
        print()
        plt.figure(figsize=(12,8))
        top_diagnoses = diagnoses_df['diagnosisname'].value_counts().head(10)
        sns.barplot(x=top_diagnoses.values, y=top_diagnoses.index, palette='magma')
        plt.title('Top 10 Most Common Diagnoses')
        plt.xlabel('Number of Occurrences')
        plt.ylabel('Diagnosis')
        plt.show()
        print()
        print("Diagnosis Distribution by Gender") #Colour decide on your own as I am not sure
        print()
        user_diagnoses_df = pd.merge(user_diagnoses_df, users_df, on='userid')
        diagnosis_gender = user_diagnoses_df.groupby(['diagnosisid', 'userid']).size().unstack(fill_value=0)
        diagnosis_gender.plot(kind='bar', stacked=True, figsize=(12,8), colormap='Accent')
        plt.title('Diagnosis Distribution by Gender')
        plt.xlabel('Diagnosis')
        plt.ylabel('Number of Occurrences')
        plt.legend(title='Gender')
        plt.show()
        print()
        print("Confidence score distribution")
        print()
        #Plotting the histogram for 'confidencescore'
        user_diagnoses_df['confidencescore'].hist(bins=10, edgecolor='black')

        # Adding title and labels
        plt.title('Histogram of Confidence Scores')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')

        # Display the plot
        plt.show()
    elif a=="Doctors" or "doctors":
        print()
        print("Doctors Based on their speciality")
        print()
        plt.figure(figsize=(10,8))
        specialty_counts = doctors_df['specialization'].value_counts()
        sns.barplot(x=specialty_counts.values, y=specialty_counts.index, palette='viridis')
        plt.title('Distribution of Doctor Specialties')
        plt.xlabel('Number of Doctors')
        plt.ylabel('Specialty')
        plt.show()
        print()
        print("User Doctor Recommendations")
        print()
        print("Error aa rha h kal krenge")
    elif a=="Medications" or "medications":
        print()
        print(" Bar Graph of Most prescribed medicines")
        print()
        plt.figure(figsize=(12,8))
        top_medications = medications_df['medicationname'].value_counts().head(10)
        sns.barplot(x=top_medications.values, y=top_medications.index, palette='plasma')
        plt.title('Top 10 Most Prescribed Medications')
        plt.xlabel('Number of Prescriptions')
        plt.ylabel('Medication')
        plt.show()
    elif a=="UserMedications" or "usermedications":
        print()
        print("Histogram for Medications per user")
        print()
        print("Yeh nhi bnaa bhai")
        print()
        print(" Line Plot for Medication use over time")
        print("Yeh bhi nhi bana")
        print()
        print("Medications by age group")
        print()
        # Merge with Users to get age
        user_medications_df = pd.merge(user_medications_df, users_df[['userid', 'age']], on='userid', how='left')
        plt.figure(figsize=(12,8))
        sns.boxplot(x='age', y='dosage', data=user_medications_df, palette='coolwarm')
        plt.title('Medication Dosage by Age Group')
        plt.xlabel('Age ')
        plt.ylabel('Dosage')
        plt.show()
    elif a==" Remedies" or "remedies":
        print()
        print("Bar Chart for most famous Remedies")
        print()
        plt.figure(figsize=(12,8))
        top_remedies = remedies_df['remedyname'].value_counts().head(10)
        sns.barplot(x=top_remedies.values, y=top_remedies.index, palette='cividis')
        plt.title('Top 10 Most Common Remedies')
        plt.xlabel('Number of Uses')
        plt.ylabel('Remedy')
        plt.show()
    elif a=="UserRemedies" or "userremedies":
        print()
        print("Histogram of users in Histogram")
        print()
        print("Error aagya bhai")
        print()
        print("Remedies Over time ")
        print()
        print("Line plot h error aayega kal krenge")
    elif a=="DietCharts" or "dietcharts":
        print()
        print("Diet Adherence over time")
        print()
        print(" Error aagya kal subah dekhenge gud night nhi bna graph")
    elif a=="HealthcareCharts" or "healthcarecharts":
        print()
        print("Yeh toh pura hi ab subah krenge Gud nyt")
    else:
        break


# In[4]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# In[8]:


plt.figure(figsize=(10,8))
specialty_counts = doctors_df['specialization'].value_counts()
sns.barplot(x=specialty_counts.values, y=specialty_counts.index, palette='viridis')
plt.title('Distribution of Doctor Specialties')
plt.xlabel('Number of Doctors')
plt.ylabel('Specialty')
plt.show()


# In[15]:


symptoms_per_user = user_symptoms_df['userid'].value_counts()
user_symptoms_df = user_symptoms_df.dropna(subset=['userid'])
user_symptoms_df['userid'] = pd.to_numeric(user_symptoms_df['userid'], errors='coerce')
user_symptoms_df = user_symptoms_df.dropna(subset=['userid'])
plt.figure(figsize=(10,6))
plt.hist(user_symptoms_df['userid'], bins=20, edgecolor='black')
plt.title('Number of Symptoms Reported per User')
plt.xlabel('Number of Symptoms')
plt.ylabel('Number of Users')
plt.show()


# In[21]:


# Assuming 'UserSymptoms' has a 'Date' column
import numpy as np
user_symptoms_df = user_symptoms_df.dropna(subset=['datereported'])
user_symptoms_df['datereported'] = pd.to_datetime(user_symptoms_df['datereported'], errors='coerce')
user_symptoms_df = user_symptoms_df.dropna(subset=['datereported'])
symptoms_over_time = user_symptoms_df.set_index('datereported').resample('W').size()


plt.figure(figsize=(12,6))
# Convert index and values to numpy arrays
x = np.array(symptoms_over_time.index)
y = np.array(symptoms_over_time.values)

# Plotting the data using plt.plot()
plt.plot(x, y, marker='o', color='purple')
plt.title('Number of Symptoms Reported Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Symptoms')
plt.show()


# In[27]:


user_doctor_recommendations_df = user_doctor_recommendations_df.dropna(subset=['recommendationdate'])
user_doctor_recommendations_df['recommendationdate'] = pd.to_datetime(user_doctor_recommendations_df['recommendationdate'], errors='coerce')
user_doctor_recommendations_df = user_doctor_recommendations_df.dropna(subset=['recommendationdate'])
recommendations_over_time = user_doctor_recommendations_df.set_index('recommendationdate').resample('M').size()

plt.figure(figsize=(12,6))
# Convert index and values to numpy arrays
x = np.array(symptoms_over_time.index)
y = np.array(symptoms_over_time.values)

# Plotting the data using plt.plot()
plt.plot(x, y, marker='o', color='brown')
plt.title('Doctor Recommendations Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Recommendations')
plt.show()


# In[30]:


user_medications_df = user_medications_df.dropna(subset=['userid'])
medications_per_user = user_medications_df['userid'].value_counts()
user_medications_df = user_medications_df.dropna(subset=['userid'])

plt.figure(figsize=(10,6))
plt.hist(user_medications_df['userid'], bins=20, color='lightgreen')
plt.title('Number of Medications per User')
plt.xlabel('Number of Medications')
plt.ylabel('Number of Users')
plt.show()


# In[33]:


# Assuming 'StartDate' and 'EndDate' exist
user_medications_df['startdate'] = pd.to_datetime(user_medications_df['startdate'], errors='coerce')
user_medications_df['enddate'] = pd.to_datetime(user_medications_df['enddate'], errors='coerce')

# Calculate duration
user_medications_df['Duration'] = (user_medications_df['enddate'] - user_medications_df['startdate']).dt.days

# Plot average duration over time
average_duration = user_medications_df.set_index('startdate').resample('W')['Duration'].mean()
x=np.array(average_duration.index)
y=np.array(average_duration.values)
plt.figure(figsize=(12,6))
plt.plot(x,y, marker='o', color='blue')
plt.title('Average Medication Adherence Duration Over Time')
plt.xlabel('Date')
plt.ylabel('Average Duration (Days)')
plt.show()


# In[35]:


remedies_per_user = user_remedies_df['userid'].value_counts()

plt.figure(figsize=(10,6))
plt.hist(remedies_per_user, bins=20, color='gold')
plt.title('Number of Remedies Used per User')
plt.xlabel('Number of Remedies')
plt.ylabel('Number of Users')
plt.show()


# In[ ]:




