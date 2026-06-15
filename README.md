# 🚗 EV Smart Range Forecaster

### 🚀 [**View the Live Demo Here!**](https://ev-energy-forecaster.streamlit.app/) 🚀

This is a complete web application that predicts Electric Vehicle (EV) energy consumption using a machine learning model and provides a generative AI assistant for user queries.

This project was built to solve the problem of EV "range anxiety." While manufacturers provide an official range, the *real-world* range is heavily affected by factors like weather, speed, and road type. This tool provides a more accurate prediction based on these real-world conditions.

## 🚀 Features

* **Machine Learning Model:** A Multi-Variate Linear Regression model trained on a real-world EV energy dataset.
    * **Achieved an R-squared (R²) of 94.61%**, far exceeding the initial 80% goal.
    * **Achieved a Mean Absolute Error (MAE) of 0.41 kWh**, making it highly accurate for real-world predictions.
* **Manual Predictor:** A clean, mobile-friendly interface built in Streamlit (using `st.expander` and `st.columns`) that allows users to manually adjust 16 different parameters and get an instant prediction from the ML model.
* **Hybrid Gen AI Chatbot:** A smart chat assistant powered by the Groq API.
    * **General Q&A:** The chatbot can answer general knowledge questions (e.g., "What is regenerative braking?").
    * **ML Model Integration:** The chatbot uses **function calling** to understand natural language queries about predictions (e.g., *"what's my usage at 90 km/h in 30c on a highway?"*).
    * **Rate-Limit Stable:** The chat logic is a **"hybrid" model**. It uses one API call to understand the user, then runs the local ML model and formats the answer *itself*, avoiding the 2-request-per-minute API limit.
    * **Professional UI:** The chat input is "paused" (disabled) and shows a spinner while the AI is processing a request, just like professional chat applications.

## 💻 Technologies Used

* **Python:** Core programming language.
* **scikit-learn:** For building and training the Linear Regression model.
* **Pandas:** For all data loading, cleaning, and preprocessing (one-hot encoding).
* **Joblib:** For saving and loading the trained ML model (`model.pkl`).
* **Streamlit:** For building and deploying the entire interactive web application.
* **Groq API:** For powering the generative AI chatbot and its function-calling capabilities.

## 🏃 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/mohit7739/EV-Energy-Forecaster.git](https://github.com/mohit7739/EV-Energy-Forecaster.git)
    cd EV-Energy-Forecaster
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

3.  **Install all dependencies:**
    (Make sure you have a `requirements.txt` file in your repository)
    ```bash
    pip install -r requirements.txt
    ```

4.  **Add your API Key:**
    * Create a new folder: `mkdir .streamlit`
    * Create a secrets file: `nano .streamlit/secrets.toml`
    * Paste your Groq API key into this file:
        ```toml
        GROQ_API_KEY = "API_KEY🤫"
        ```
    * Save and close the file. (This file is ignored by `.gitignore` so it will not be uploaded).

5.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    Your browser will automatically open to the application.