# pip install nltk
# pip install ssl
# pip install streamlit
# pip install scikit-learn

import os
import nltk

import ssl
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')


intents = [
    {
        "tag": "greeting",  
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.", "Age is just a number for me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": ["To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.", "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.", "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": ["A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.", "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame."]
    },
    {
        "tag": "product_info",
        "patterns": ["Tell me about [product]", "What are the features of [product]", "Can you provide information on [product]"],
        "responses": ["[Product] is a [description of product]. Its features include [list of features].", "Sure, [product] is known for [highlighted feature] and [another highlighted feature].", "I'd be happy to help you learn more about [product]."]
    },
    {
        "tag": "tech_support",
        "patterns": ["I'm having trouble with [issue]", "How do I fix [issue]", "Help me troubleshoot [issue]"],
        "responses": ["Let's see if we can solve that. Have you tried [common solution]?","For [issue], a common solution is to [action]. Have you attempted that yet?", "I'm here to assist you with [issue]. Let's work through it together."]
    },
    {
        "tag": "booking_assistance",
        "patterns": ["How do I book [service]", "Can you help me with my reservation", "I need assistance with booking [service]"],
        "responses": ["Certainly! To book [service], you can visit our website or contact our customer service team at [phone number].", "I can guide you through the booking process. What [service] are you looking to book?", "Let's get started on your reservation. What dates are you considering for [service]?"]
    },
    {
        "tag": "education_resources",
        "patterns": ["Where can I find learning resources", "Do you have any educational materials", "I want to learn about [topic]"],
        "responses": ["There are various online platforms like Coursera, Khan Academy, and Udemy where you can find courses on [topic].", "I can suggest some books and websites for learning about [topic]. Would you like some recommendations?", "Learning about [topic] is a great idea! I can provide you with some resources to get started."]
    },
    {
        "tag": "health_tips",
        "patterns": ["How can I stay healthy", "Give me some wellness advice", "I want to improve my health"],
        "responses": ["Staying hydrated, eating balanced meals, and getting regular exercise are key to maintaining good health.", "Incorporating mindfulness practices like meditation can also promote overall wellness.", "It's important to prioritize self-care and listen to your body's needs. Small changes like getting enough sleep and managing stress can make a big difference."]
    },
    {
        "tag": "restaurant_recommendation",
        "patterns": ["Can you recommend a good restaurant?", "Where should I eat?", "Suggest a restaurant", "Best places to eat around me"],
        "responses": ["I recommend trying [Restaurant Name]. It has great reviews!", "You should check out [Restaurant Name] for a delicious meal.", "How about dining at [Restaurant Name]? It's quite popular."]
    },
    {
        "tag": "movie_recommendation",
        "patterns": ["What movie should I watch?", "Recommend a good movie", "Suggest a film", "Any good movies to watch?"],
        "responses": ["I suggest watching [Movie Name]. It's a great choice!", "How about [Movie Name]? It's highly rated.", "You might enjoy [Movie Name]. It's quite popular."]
    },
    {
        "tag": "music_recommendation",
        "patterns": ["Can you suggest some music?", "Recommend a song", "What's a good album to listen to?", "Any music recommendations?"],
        "responses": ["You should listen to [Song/Album Name] by [Artist].", "I recommend [Song/Album Name] by [Artist]. It's really good.", "How about trying [Song/Album Name]? It's quite popular."]
    },
    {
        "tag": "news_update",
        "patterns": ["What's the latest news?", "Update me on current events", "Any news updates?", "Tell me what's happening in the world"],
        "responses": ["Here are today's top headlines: [News Headline 1], [News Headline 2], [News Headline 3].", "Here's what's happening: [News Summary].", "Today's news includes: [News Headline]."]
    },
    {
        "tag": "sports_update",
        "patterns": ["What's the latest in sports?", "Give me a sports update", "Any sports news?", "Tell me about the recent game"],
        "responses": ["In sports today: [Sports News Headline].", "Here's the latest sports update: [Sports News Summary].", "Today's sports highlights include: [Sports Highlight]."]
    },
    {
        "tag": "recipe_suggestion",
        "patterns": ["Can you suggest a recipe?", "What should I cook?", "Give me a recipe idea", "Any good recipes?"],
        "responses": ["How about making [Dish Name]? Here's the recipe: [Recipe Link].", "You could try cooking [Dish Name]. It's delicious!", "I recommend [Dish Name]. It's a great dish."]
    },
    {
        "tag": "exercise_tips",
        "patterns": ["Give me some exercise tips", "How can I stay fit?", "Suggest a workout routine", "Any fitness advice?"],
        "responses": ["You can try a mix of cardio and strength training for a balanced workout.", "I suggest starting with a daily walk and some light weights.", "Try doing a 30-minute workout every day, mixing in cardio and stretching."]
    },
    {
        "tag": "travel_tips",
        "patterns": ["Any travel tips?", "How should I prepare for a trip?", "Give me travel advice", "What should I know before traveling?"],
        "responses": ["Always pack light and carry essentials in your hand luggage.", "Make sure to research your destination and plan ahead.", "It's a good idea to have a copy of important documents and travel insurance."]
    },
    {
        "tag": "local_events",
        "patterns": ["What's happening in my city?", "Any local events?", "Tell me about events nearby", "What's going on locally?"],
        "responses": ["There is a [Event Name] happening at [Location] on [Date].", "Check out the [Event Name] at [Location].", "This weekend, there's a [Event Name] you might enjoy."]
    },
    {
        "tag": "emergency_contact",
        "patterns": ["What's the emergency number?", "How do I contact emergency services?", "Give me the number for emergency services", "Who do I call in an emergency?"],
        "responses": ["In case of an emergency, call 911.", "For emergencies, dial 911 immediately.", "You should call 911 for emergency services."]
    },
    {
        "tag": "language_translation",
        "patterns": ["How do you say [phrase] in [language]?", "Translate [phrase] to [language]", "What is [phrase] in [language]?", "Can you translate [phrase] for me?"],
        "responses": ["[Phrase] in [Language] is [Translation].", "You say [Translation] in [Language].", "In [Language], you would say [Translation]."]
    },
    {
        "tag": "time_conversion",
        "patterns": ["What time is it in [City]?", "Convert [time] to [City] time", "What’s the current time in [City]?", "Time in [City] now"],
        "responses": ["The current time in [City] is [Time].", "In [City], it is currently [Time].", "[Time] is the local time in [City] now."]
    },
    {
        "tag": "currency_conversion",
        "patterns": ["What's the exchange rate from [Currency1] to [Currency2]?", "Convert [Amount] [Currency1] to [Currency2]", "How much is [Amount] [Currency1] in [Currency2]?", "Exchange rate for [Currency1] to [Currency2]"],
        "responses": ["The exchange rate from [Currency1] to [Currency2] is [Rate].", "[Amount] [Currency1] is approximately [Converted Amount] [Currency2].", "Currently, [Amount] [Currency1] equals [Converted Amount] [Currency2]."]
    },
    {
        "tag": "public_transport_info",
        "patterns": ["What’s the next bus to [Location]?", "Public transport schedule to [Location]", "When is the next train to [Location]?", "Bus timings to [Location]"],
        "responses": ["The next bus to [Location] is at [Time].", "You can catch the next train to [Location] at [Time].", "The public transport schedule to [Location] shows the next departure at [Time]."]
    },
    {
        "tag": "shopping_recommendation",
        "patterns": ["Where can I buy [product]?", "Recommend a store for [product]", "Best place to shop for [product]", "Where should I purchase [product]?"],
        "responses": ["You can buy [product] at [Store Name].", "I recommend shopping at [Store Name] for [product].", "Check out [Store Name] for the best [product]."]
    },
    {
        "tag": "appointment_booking",
        "patterns": ["How do I book an appointment?", "Schedule a meeting with [Person/Service]", "Book an appointment for [Service]", "How can I make an appointment?"],
        "responses": ["You can book an appointment by calling [Phone Number] or visiting [Website].", "To schedule a meeting, contact [Person/Service] through [Contact Information].", "Appointments for [Service] can be booked online at [Website] or by phone at [Phone Number]."]
    },
    {
        "tag": "payment_methods",
        "patterns": ["What payment methods do you accept?", "Can I pay with [Payment Method]?", "Do you accept [Payment Method]?", "How can I pay for [Service/Product]?"],
        "responses": ["We accept various payment methods including [Payment Methods].", "You can pay using [Payment Methods].", "[Payment Method] is accepted here along with others like [Other Payment Methods]."]
    },
    {
        "tag": "return_policy",
        "patterns": ["What's your return policy?", "Can I return [Product]?", "How do I return an item?", "What are the conditions for returning [Product]?"],
        "responses": ["Our return policy allows returns within [Number] days with a receipt.", "You can return [Product] within [Number] days of purchase.", "Items can be returned under these conditions: [Return Conditions]."]
    },
    {
        "tag": "holiday_greetings",
        "patterns": ["Merry Christmas!", "Happy New Year!", "Happy Holidays!", "Season's Greetings!"],
        "responses": ["Merry Christmas to you too!", "Happy New Year! Wishing you all the best.", "Happy Holidays! Enjoy the festive season.", "Season's Greetings!"]
    },
    {
        "tag": "covid_guidelines",
        "patterns": ["What are the current COVID-19 guidelines?", "COVID-19 safety measures", "How to stay safe from COVID?", "COVID-19 travel restrictions"],
        "responses": ["Current COVID-19 guidelines include wearing masks, social distancing, and regular hand washing.", "To stay safe, follow these measures: [Safety Measures].", "COVID-19 travel restrictions vary by location. Check local guidelines before traveling."]
    },
    {
        "tag": "career_advice",
        "patterns": ["Can you give me career advice?", "How do I advance in my career?", "Tips for a successful career", "How to achieve my career goals?"],
        "responses": ["Focus on continuous learning and skill development.", "Networking and seeking mentorship can greatly help your career.", "Set clear, achievable goals and work consistently towards them."]
    }
]

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        

counter = 0

def main():
    global counter
    st.title("Chatbot")
    st.write("Welcome to the chatbot. Please type a message and press Enter")
    
    counter += 1
    user_input = st.text_input("You:", key=f"user_input_{counter}")
    
    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response,height=100, max_chars=None)
        
        if response.lower() in ['goodbye','bye']:
            st.write("Thank you for chatting with me. Have a great day! ")
            st.stop()
            
if __name__ == '__main__':
    main()