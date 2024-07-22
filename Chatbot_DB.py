import pymysql as sql
import random
import os
import nltk
import ssl
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Function to establish database connection
def conn():
    srvr = sql.connect(user='root', password='', host='localhost', port=3306, database='chatbot')
    crsr = srvr.cursor()
    return srvr, crsr

# Setup NLTK data and SSL context
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Global variable to store fetched data
data1 = []

# Function to fetch data from the database
def fetch_data():
    global data1
    srvr, cur = conn()
    cmd = "SELECT * FROM intents;"
    cur.execute(cmd)
    data1 = cur.fetchall()
    srvr.close()  # Close the connection after fetching data

# Function to fetch existing tags from the database
def fetch_existing_tags():
    srvr, cur = conn()
    cmd = "SELECT tag FROM intents;"
    cur.execute(cmd)
    tags = cur.fetchall()
    srvr.close()
    return [tag[0] for tag in tags]

# Function to insert new intent into the database with unique tag
def insert_new_intent(pattern, response):
    srvr, cur = conn()
    existing_tags = fetch_existing_tags()
    
    # Generate unique tag
    tag_num = 1
    while f'new_intent_{tag_num}' in existing_tags:
        tag_num += 1
    tag = f'new_intent_{tag_num}'
    
    cur.execute("INSERT INTO intents (tag, patterns, responses) VALUES (%s, %s, %s)", 
                (tag, pattern, response))
    srvr.commit()
    srvr.close()

# Function to check if input exists in the database
def check_existing_intent(input_text):
    for row in data1:
        if input_text in row[2]:
            responses = row[3]
            response_list = responses.split('|')
            return random.choice(response_list), True
    return None, False

# Function to train the model
def train_model():
    global data1
    patterns = [row[2] for row in data1]
    tags = [row[1] for row in data1]
    x = vectorizer.fit_transform(patterns)
    clf.fit(x, tags)

# Web scraping function
def scrapping(input_text):
    path = "C:/Users/kulde/Downloads/chromedriver-win64/chromedriver-win64/chromedriver.exe"
    s = Service(path)
    driver = webdriver.Chrome(service=s)
    
    try:
        driver.get("https://google.com/")
        time.sleep(3)
        
        box = driver.find_element(By.XPATH, "/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/textarea")
        box.send_keys(input_text + " wikipedia")
        box.send_keys(Keys.ENTER)
        
        time.sleep(3)  # Wait for the page to load
        
        results = driver.find_elements(By.XPATH, "//*/span/span")        
        result_texts = [result.text for result in results if result.text.strip() != '']
        
    except Exception as e:
        print(f"An error occurred: {e}")
        result_texts = ["Sorry, I couldn't find an answer for that."]

    return result_texts

# Chatbot function to get a response
def chatbot(input_text):
    global data1
    if not data1:
        fetch_data()
        train_model()

    existing_response, found_in_db = check_existing_intent(input_text)
    
    if existing_response:
        return existing_response
    
    # If input is not found in the database, use web scraping to find an answer
    if not found_in_db:
        scraped_responses = scrapping(input_text)
        if scraped_responses:
            response = scraped_responses[0]
        else:
            response = "I am still in learning phase so you can about this on Google, Thanks 😊"
        
        # Add the new intent to the database
        insert_new_intent(input_text, response)
        fetch_data()  # Fetch updated data
        train_model()  # Retrain the model with updated data
        
        return response
    
    return "I am still in learning phase so you can about this on Google, Thanks 😊"

# Counter for unique text input keys in Streamlit
counter = 0

# Main function for Streamlit app
def main():
    global counter
    st.title("Chatbot")
    st.write("Welcome to the chatbot. Please type a message and press Enter")

    counter += 1
    user_input = st.text_input("You:", key=f"user_input_{counter}")

    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None)

        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

if __name__ == '__main__':
    main()
