import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyBdEe_Auy7zsRrWLYHsU5cPwXTBCG9V7eY")

# Define study-related keywords
STUDY_KEYWORDS = ["exam", "syllabus", "preparation", "study", "topic", "books",
                  "revision", "concept", "question", "university", "test", 
                  "marks", "rank", "cutoff", "subject", "notes", "strategy", "gate","gre"]

# Create Gemini model
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

chat_session = model.start_chat(history=[])

# Function to check if the query is study-related
def is_study_related(user_input):
    user_words = user_input.lower().split()
    return any(keyword in user_words for keyword in STUDY_KEYWORDS)

# Chatbot interaction loop
print("STUDY-BOT: Hello! Ask me about study topics, exams, and preparation. Type 'exit' to stop.")

while True:
    try:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("STUDY-BOT: Goodbye! Keep studying hard! 📚✨")
            break
        
        # Check if the query is study-related
        if is_study_related(user_input):
            response = chat_session.send_message(user_input)
            print("STUDY-BOT:", response.text)
        else:
            print("STUDY-BOT: I only answer study-related queries. Please ask about exams, topics, or preparation.")
    
    except KeyboardInterrupt:
        print("\nSTUDY-BOT: Exiting chat...")
        break
    except Exception as e:
        print(f"STUDY-BOT: An error occurred: {str(e)}")
        continue