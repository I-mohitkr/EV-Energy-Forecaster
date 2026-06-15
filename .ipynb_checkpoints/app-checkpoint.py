import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, Tool, FunctionDeclaration
import google.generativeai.protos as protos # For older library compatibility

# --- Page Setup (Aesthetic & Mobile-Friendly) ---
st.set_page_config(
    page_title="EV Smart Range Forecaster",
    page_icon="🚗",
    layout="centered", # Centered layout is better for mobile
    initial_sidebar_state="collapsed" # Start with sidebar collapsed
)

# --- Cached Functions to Load Models and API ---

@st.cache_data
def load_models():
    """Loads the saved ML model and column list from disk."""
    try:
        model = joblib.load('model.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, model_columns
    except FileNotFoundError:
        st.error("Model files not found! Please run the notebook to generate 'model.pkl' and 'model_columns.pkl'.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred loading the model: {e}")
        return None, None

@st.cache_resource
def load_genai_model():
    """Connects to the Gemini API and returns the model, configured with tools."""
    try:
        API_KEY = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=API_KEY)
        
        # Define the function for the AI to use
        predict_energy_tool = FunctionDeclaration(
            name="predict_energy_consumption",
            description="Predicts the EV energy consumption in kWh based on vehicle and environmental factors. Use this when a user asks for a prediction, even if they only provide some of the factors.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "speed_kmh": {"type": "NUMBER", "description": "Vehicle speed in km/h"},
                    "temperature_c": {"type": "NUMBER", "description": "Ambient temperature in Celsius"},
                    "slope_percent": {"type": "NUMBER", "description": "Road slope in percent (e.g., 0.0)"},
                    "driving_mode": {"type": "INTEGER", "description": "Driving mode (1: Eco, 2: Normal, 3: Sport)"},
                    "road_type": {"type": "INTEGER", "description": "Road type (1: Highway, 2: Urban, 3: Rural)"},
                    "traffic_condition": {"type": "INTEGER", "description": "Traffic condition (1: Low, 2: Medium, 3: High)"},
                    "weather_condition": {"type": "INTEGER", "description": "Weather (1: Sunny, 2: Rainy, 3: Cloudy, 4: Snowy)"},
                },
                "required": []
            }
        )
        
        # "AUTO" lets the AI decide if the prompt is a chat or a function call.
        tool_config = {
            "function_calling_config": {
                "mode": "AUTO" 
            }
        }
        
        # Create the model and pass BOTH the tool AND the tool_config
        genai_model = genai.GenerativeModel(
            model_name="models/gemini-pro-latest",
            tools=[predict_energy_tool],
            tool_config=tool_config,
            generation_config=GenerationConfig(temperature=0.1) # Set low temp for reliable function calling
        )
        return genai_model
    except Exception as e:
        st.error(f"Error connecting to Gemini API: {e}. Check your API key in .streamlit/secrets.toml")
        return None

# --- Local Python Function to run the ML model ---
# --- THIS IS THE FIX ---
# The function definition is updated to accept ALL possible arguments from the manual sliders
# or the chatbot.
def predict_energy(model, model_columns, 
                   speed_kmh=None, temperature_c=None, slope_percent=None, 
                   driving_mode=None, road_type=None, traffic_condition=None, weather_condition=None,
                   acceleration_ms2=None, battery_state_percent=None, battery_voltage_v=None,
                   battery_temp_c=None, humidity_percent=None, wind_speed_ms=None,
                   tire_pressure_psi=None, vehicle_weight_kg=None, distance_travelled_km=None):
    """
    This is the actual Python function that runs your saved scikit-learn model.
    It now accepts all 16 features and uses a default if one isn't provided.
    """
    
    # Create a dictionary for all inputs, using provided values or defaults
    input_data = {
        'Speed_kmh': speed_kmh if speed_kmh is not None else 60.0,
        'Acceleration_ms2': acceleration_ms2 if acceleration_ms2 is not None else 0.0,
        'Battery_State_%': battery_state_percent if battery_state_percent is not None else 80.0,
        'Battery_Voltage_V': battery_voltage_v if battery_voltage_v is not None else 350.0,
        'Battery_Temperature_C': battery_temp_c if battery_temp_c is not None else 25.0,
        'Slope_%': slope_percent if slope_percent is not None else 0.0,
        'Temperature_C': temperature_c if temperature_c is not None else 20.0,
        'Humidity_%': humidity_percent if humidity_percent is not None else 50.0,
        'Wind_Speed_ms': wind_speed_ms if wind_speed_ms is not None else 5.0,
        'Tire_Pressure_psi': tire_pressure_psi if tire_pressure_psi is not None else 32.0,
        'Vehicle_Weight_kg': vehicle_weight_kg if vehicle_weight_kg is not None else 1800.0,
        'Distance_Travelled_km': distance_travelled_km if distance_travelled_km is not None else 10.0,
        'Driving_Mode': driving_mode if driving_mode is not None else 1,
        'Road_Type': road_type if road_type is not None else 1,
        'Traffic_Condition': traffic_condition if traffic_condition is not None else 1,
        'Weather_Condition': weather_condition if weather_condition is not None else 1
    }
    
    # Convert to DataFrame, one-hot encode, and align columns
    input_df = pd.DataFrame([input_data])
    categorical_cols = ['Driving_Mode', 'Road_Type', 'Traffic_Condition', 'Weather_Condition']
    input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols, prefix=categorical_cols)
    input_df_aligned = input_df_encoded.reindex(columns=model_columns, fill_value=0)
    
    # Make the prediction
    prediction = model.predict(input_df_aligned)
    return prediction[0]


# --- Load Models and API ---
model, model_columns = load_models()
genai_model = load_genai_model()

# Check if models loaded, but use subtle notifications
if not model or not model_columns: 
    st.error("ML Model is not loaded. Manual prediction is disabled.")
    st.stop()
    
if not genai_model: 
    st.error("Chatbot is not connected. Please check your API key.")
    st.stop()

# Use st.toast for a cleaner, professional notification
# This will show up once in the bottom-right corner and fade away.
st.toast("Models and Chatbot loaded successfully!", icon="✅")


@st.cache_resource
def start_chat_session(_genai_model):
    return _genai_model.start_chat(enable_automatic_function_calling=False)

chat = start_chat_session(genai_model)


# --- Main Page Layout ---

st.title("🚗 EV Smart Range Forecaster")

# --- 1. Manual Prediction (Moved to Expander) ---
with st.expander("Show Manual Prediction Controls"):
    
    st.write("Use the controls below to get a prediction from the 94.6% accurate ML model.")
    
    # Use columns for a cleaner layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Driving")
        speed_kmh_slider = st.slider("Speed (km/h)", 0.0, 120.0, 60.0)
        slope_percent_slider = st.slider("Road Slope (%)", -5.0, 10.0, 0.0)
        driving_mode_slider = st.selectbox("Driving Mode", [1, 2, 3], 
                                           format_func=lambda x: {1: 'Eco', 2: 'Normal', 3: 'Sport'}.get(x))
        road_type_slider = st.selectbox("Road Type", [1, 2, 3], 
                                        format_func=lambda x: {1: 'Highway', 2: 'Urban', 3: 'Rural'}.get(x))
        traffic_condition_slider = st.selectbox("Traffic Condition", [1, 2, 3], 
                                                format_func=lambda x: {1: 'Low', 2: 'Medium', 3: 'High'}.get(x))
    
    with col2:
        st.subheader("Environment")
        temperature_c_slider = st.slider("Temperature (°C)", -10.0, 45.0, 20.0)
        weather_condition_slider = st.selectbox("Weather Condition", [1, 2, 3, 4], 
                                                format_func=lambda x: {1: 'Sunny', 2: 'Rainy', 3: 'Cloudy', 4: 'Snowy'}.get(x))
        wind_speed_ms_slider = st.slider("Wind Speed (m/s)", 0.0, 15.0, 5.0)
        humidity_percent_slider = st.slider("Humidity (%)", 20.0, 90.0, 50.0)
    
    st.subheader("Other (Optional)")
    # Put less important sliders in more columns to save space
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        acceleration_ms2_slider = st.slider("Acceleration (m/s²)", -3.0, 3.0, 0.0)
        battery_state_percent_slider = st.slider("Battery State (%)", 20.0, 100.0, 80.0)
    with col_b:
        battery_voltage_v_slider = st.slider("Battery Voltage (V)", 300.0, 400.0, 350.0)
        battery_temp_c_slider = st.slider("Battery Temp (°C)", 10.0, 45.0, 25.0)
    with col_c:
        tire_pressure_psi_slider = st.slider("Tire Pressure (psi)", 28.0, 35.0, 32.0)
        vehicle_weight_kg_slider = st.slider("Vehicle Weight (kg)", 1200.0, 2500.0, 1800.0)
    
    distance_travelled_km_slider = st.slider("Distance Travelled (km)", 0.0, 50.0, 10.0)


    # Create a button in the main area (outside the expander)
    if st.button("Predict Manually"):
        if model:
            # Run the prediction logic
            # --- THIS IS THE FIX ---
            # Now the function call passes all 16 arguments
            prediction = predict_energy(
                model=model, model_columns=model_columns,
                speed_kmh=speed_kmh_slider, 
                temperature_c=temperature_c_slider, 
                slope_percent=slope_percent_slider, 
                driving_mode=driving_mode_slider, 
                road_type=road_type_slider, 
                traffic_condition=traffic_condition_slider, 
                weather_condition=weather_condition_slider,
                acceleration_ms2=acceleration_ms2_slider, 
                battery_state_percent=battery_state_percent_slider,
                battery_voltage_v=battery_voltage_v_slider, 
                battery_temp_c=battery_temp_c_slider,
                humidity_percent=humidity_percent_slider, 
                wind_speed_ms=wind_speed_ms_slider,
                tire_pressure_psi=tire_pressure_psi_slider, 
                vehicle_weight_kg=vehicle_weight_kg_slider,
                distance_travelled_km=distance_travelled_km_slider
            )
            
            st.metric(label="Predicted Energy Consumption", value=f"{prediction:.2f} kWh")
        else:
            st.error("Model is not loaded. Cannot make prediction.")

# --- DIVIDER ---
st.divider()

# --- GEN AI CHATBOT (Hybrid, Stable Version) ---
st.header("🤖 EV Chat Assistant")
st.info("Ask general EV questions or get a prediction! (e.g., 'what's my usage at 90 km/h in 30c on a highway?')")

# Initialize chat history and processing state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False # To pause input

# Display past chat messages
if "messages" in st.session_state:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Get new user input, disable if processing
if prompt := st.chat_input("Ask a general question or for a prediction...", disabled=st.session_state.processing): 
    if not genai_model:
        st.error("Chatbot is not connected. Cannot send message.")
        st.stop()

    # --- Start processing, disable input ---
    st.session_state.processing = True 
    
    # Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show a spinner while the AI is thinking
    with st.spinner("🤖 AI is thinking..."):
        ai_response_text = ""
        try:
            # --- API CALL #1 (This is the *only* API call) ---
            response1 = chat.send_message(prompt)
            
            # Check for function call
            part = response1.candidates[0].content.parts[0]
            
            if part.function_call and part.function_call.name == "predict_energy_consumption":
                args = part.function_call.args
                
                # --- NEW CHECK: See if the AI actually extracted any arguments ---
                if not args:
                    # Case: AI wanted to call the function but didn't find any parameters
                    ai_response_text = "I can definitely help with that! Please tell me a bit more, like the speed, weather, or road type you're interested in."
                else:
                    # Case: AI found parameters, let's run the model!
                    
                    # Call our *local* Python function
                    # --- THIS IS THE FIX ---
                    # Now the chatbot call also uses the full function, passing None for missing args
                    prediction_result = predict_energy(
                        model=model,
                        model_columns=model_columns,
                        speed_kmh=args.get("speed_kmh"), # Send None if missing
                        temperature_c=args.get("temperature_c"), # Send None if missing
                        slope_percent=args.get("slope_percent"), # Send None if missing
                        driving_mode=args.get("driving_mode"), # Send None if missing
                        road_type=args.get("road_type"), # Send None if missing
                        traffic_condition=args.get("traffic_condition"), # Send None if missing
                        weather_condition=args.get("weather_condition") # Send None if missing
                    )

                    # --- NO API CALL #2 ---
                    # We format the answer ourselves.
                    ai_response_text = f"Based on your ML model, the predicted energy consumption for those conditions is: **{prediction_result:.2f} kWh**"
            
            else:
                # Case: Normal text response (no function call)
                ai_response_text = part.text

        except Exception as e:
            # Case: Handle ALL errors (like 429 on the FIRST call)
            ai_response_text = f"An error occurred: {e}"
            st.error(ai_response_text) # Display error in the chat
        
        # Display the final AI response
        if ai_response_text:
            with st.chat_message("assistant"):
                st.markdown(ai_response_text)
            st.session_state.messages.append({"role": "assistant", "content": ai_response_text})
        else:
            # This is a fallback
            st.error("An unknown error occurred. No response text was generated.")

    # --- End processing, re-enable input ---
    st.session_state.processing = False 
    st.rerun() # Rerun the script to reflect the new state