import re

with open("app.py", "r") as f:
    code = f.read()

# Replace imports
code = code.replace("import google.generativeai as genai\nfrom google.generativeai.types import GenerationConfig, Tool, FunctionDeclaration\nimport google.generativeai.protos as protos # For older library compatibility", "import json\nfrom groq import Groq")

# Replace load_genai_model
old_genai_model = '''@st.cache_resource
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
        return None'''

new_groq_client = '''@st.cache_resource
def load_groq_client():
    """Connects to the Groq API and returns the client."""
    try:
        API_KEY = st.secrets["GROQ_API_KEY"]
        return Groq(api_key=API_KEY)
    except Exception as e:
        st.error(f"Error connecting to Groq API: {e}. Check your API key in .streamlit/secrets.toml")
        return None'''

code = code.replace(old_genai_model, new_groq_client)

# Update the load calls
code = code.replace("genai_model = load_genai_model()", "groq_client = load_groq_client()")
code = code.replace("if not genai_model:", "if not groq_client:")

# Remove chat session init
chat_session_code = '''@st.cache_resource
def start_chat_session(_genai_model):
    return _genai_model.start_chat(enable_automatic_function_calling=False)

chat = start_chat_session(genai_model)'''
code = code.replace(chat_session_code, "")

# Replace the chat processing part
old_chat_block = '''    # Show a spinner while the AI is thinking
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
            st.error("An unknown error occurred. No response text was generated.")'''

new_chat_block = '''    # Show a spinner while the AI is thinking
    with st.spinner("🤖 AI is thinking..."):
        ai_response_text = ""
        try:
            # Setup tool
            predict_energy_tool = {
                "type": "function",
                "function": {
                    "name": "predict_energy_consumption",
                    "description": "Predicts the EV energy consumption in kWh based on vehicle and environmental factors.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "speed_kmh": {"type": "number", "description": "Vehicle speed in km/h"},
                            "temperature_c": {"type": "number", "description": "Ambient temperature in Celsius"},
                            "slope_percent": {"type": "number", "description": "Road slope in percent (e.g., 0.0)"},
                            "driving_mode": {"type": "integer", "description": "Driving mode (1: Eco, 2: Normal, 3: Sport)"},
                            "road_type": {"type": "integer", "description": "Road type (1: Highway, 2: Urban, 3: Rural)"},
                            "traffic_condition": {"type": "integer", "description": "Traffic condition (1: Low, 2: Medium, 3: High)"},
                            "weather_condition": {"type": "integer", "description": "Weather (1: Sunny, 2: Rainy, 3: Cloudy, 4: Snowy)"},
                        },
                    }
                }
            }
            
            # Prepare messages for Groq
            groq_messages = [{"role": "system", "content": "You are a helpful EV assistant. Answer questions or use the predict_energy_consumption tool if the user asks for a range/energy prediction."}]
            for msg in st.session_state.messages:
                # Groq roles are user/assistant/system
                groq_messages.append({"role": msg["role"], "content": msg["content"]})
            
            response = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=groq_messages,
                tools=[predict_energy_tool],
                tool_choice="auto",
                temperature=0.1
            )
            
            response_message = response.choices[0].message
            
            if response_message.tool_calls:
                tool_call = response_message.tool_calls[0]
                args = json.loads(tool_call.function.arguments)
                
                if not args:
                    ai_response_text = "I can definitely help with that! Please tell me a bit more, like the speed, weather, or road type you're interested in."
                else:
                    prediction_result = predict_energy(
                        model=model,
                        model_columns=model_columns,
                        speed_kmh=args.get("speed_kmh"),
                        temperature_c=args.get("temperature_c"),
                        slope_percent=args.get("slope_percent"),
                        driving_mode=args.get("driving_mode"),
                        road_type=args.get("road_type"),
                        traffic_condition=args.get("traffic_condition"),
                        weather_condition=args.get("weather_condition")
                    )
                    ai_response_text = f"Based on your ML model, the predicted energy consumption for those conditions is: **{prediction_result:.2f} kWh**"
            else:
                ai_response_text = response_message.content

        except Exception as e:
            ai_response_text = f"An error occurred: {e}"
            st.error(ai_response_text)
        
        if ai_response_text:
            with st.chat_message("assistant"):
                st.markdown(ai_response_text)
            st.session_state.messages.append({"role": "assistant", "content": ai_response_text})
        else:
            st.error("An unknown error occurred. No response text was generated.")'''

code = code.replace(old_chat_block, new_chat_block)
code = code.replace("if not genai_model:", "if not groq_client:")

with open("app.py", "w") as f:
    f.write(code)

