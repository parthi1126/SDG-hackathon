import os
import google.generativeai as genai

genai.configure(api_key="AIzaSyBdEe_Auy7zsRrWLYHsU5cPwXTBCG9V7eY")

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)

chat_session = model.start_chat(
  history=[
  ]
)
a=True
while a:
    try:
        x = input()
        if x.lower() == "exit":
            print("Exiting chat...")
            a = False
            break
        response = chat_session.send_message(x)
        print(response.text)
    except KeyboardInterrupt:
        print("\nExiting chat...")
        a = False
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        continue