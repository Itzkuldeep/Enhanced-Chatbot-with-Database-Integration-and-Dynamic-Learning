```markdown
# Chatbot with Database and Web Scraping Integration

This repository contains a Python-based chatbot project that integrates with a MySQL database and uses web scraping for information retrieval. The chatbot is implemented using Streamlit for the front-end interface, Selenium for web scraping, and scikit-learn for natural language processing (NLP).

## Contents

- [Overview](#overview)
- [Setup Instructions](#setup-instructions)
- [Running the Chatbot](#running-the-chatbot)
- [Customizing the Intents](#customizing-the-intents)
- [Dependencies](#dependencies)

## Overview

### `Chatbot_DB.py`

The `Chatbot_DB.py` file implements a chatbot that connects to a MySQL database to fetch intents and responses. It uses scikit-learn for TF-IDF vectorization and Logistic Regression for intent classification. If an input is not found in the database, the chatbot uses web scraping to find an answer on Google and updates the database with new intents.

### `Chatbot_intnt.py`

The `Chatbot_intnt.py` file contains the code for the Chatbot with Intents it have stored intents in it only and it use them for giving reponses to user, it it limited to the intents stored in it.

## Setup Instructions

### Prerequisites

- Python 3.6 or higher
- MySQL server
- ChromeDriver (for Selenium)

### Installing Dependencies

You can install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### Setting Up the MySQL Database

1. Create a MySQL database named `chatbot`.
2. Create a table named `intents` with columns for `id`, `tag`, `patterns`, and `responses`.
3. Populate the `intents` table with your data.

## Running the Chatbot

Ensure your MySQL server is running, and the `chatbot` database is set up. Then, run the Streamlit app:

```bash
streamlit run app.py
```

The app will launch in your default web browser. You can interact with the chatbot by typing messages and pressing Enter.

## Customizing the Intents

To customize the intents, you can add, edit, or delete entries directly in the `intents` table in your MySQL database.

### Adding New Intents

If the chatbot receives an input not in the database, it will use web scraping to find an answer and add the new intent to the database.

## Dependencies

The required Python packages are listed in `requirements.txt`. Key dependencies include:

- `pymysql`
- `nltk`
- `streamlit`
- `scikit-learn`
- `selenium`
- `beautifulsoup4`

Ensure all dependencies are installed by running:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
