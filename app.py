import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

# Configure Gemini API
genai.configure(api_key="AIzaSyBdEe_Auy7zsRrWLYHsU5cPwXTBCG9V7eY")

# Define study-related keywords
STUDY_KEYWORDS = ["Hi","exam", "syllabus", "preparation", "study", "topic", "books","normalization","ai"
                  "revision", "concept", "question", "university", "test", 
                  "marks", "rank", "cutoff", "subject", "notes", "strategy", "gate", "gre","GATE Exam", 
                "GATE Syllabus", "GATE Preparation","Study Material", "GATE Topics", "GATE Books", "Revision Notes", 
                "Conceptual Clarity", "Previous Year Papers", "Mock Tests", "Core Subjects","Verbal Reasoning", 
                "Analytical Reasoning", "Multiple Choice Questions (MCQs)", "Engineering Mathematics", "Mechanical Engineering (ME)", 
                "Electrical Engineering (EE)", "Civil Engineering (CE)", "Electronics and Communication Engineering (ECE)", 
                "Computer Science and IT (CS/IT)", "Chemical Engineering (CH)","Data Structures", "Algorithms", "Time Complexity", 
                "Space Complexity", "Binary Search", "Dynamic Programming", "Greedy Algorithms", "Graph Theory", "Trees", "Linked List",
                "Queue", "Stack", "Sorting Algorithms", "Recursion", "Hashing", "Heap", "Priority Queue", "Divide and Conquer", 
                "Binary Tree", "AVL Tree", "Red-Black Tree", "Dijkstra's Algorithm", "Floyd-Warshall Algorithm", "Knapsack Problem", 
                "Subset Sum", "Minimum Spanning Tree", "Topological Sort", "Breadth-First Search", "Depth-First Search", 
                "Graph Traversal", "Shortest Path", "Network Flow", "Maximum Flow", "Minimum Cut", "Matrix Multiplication", 
                "Bellman-Ford Algorithm", "Kruskal's Algorithm", "Prim's Algorithm", "Fibonacci Series", "Backtracking", 
                "Binomial Coefficients", "NP-Complete Problems", "NP-Hard Problems", "Big-O Notation", "Big-Theta Notation", 
                "Big-Omega Notation", "Space-Time Tradeoffs", "Greedy Method", "Branch and Bound", "A* Search", "Fuzzy Logic", 
                "Probabilistic Models", "Markov Chains", "Bayesian Networks", "Monte Carlo Simulation", "Artificial Intelligence", 
                "Machine Learning", "Deep Learning", "Neural Networks", "Convolutional Neural Networks", "Recurrent Neural Networks", 
                "Natural Language Processing", "Speech Recognition", "Computer Vision", "Reinforcement Learning", "Supervised Learning",
                "Unsupervised Learning", "Semi-supervised Learning", "Transfer Learning", "Generative Models", "Discriminative Models",
                "Loss Functions", "Gradient Descent", "Stochastic Gradient Descent", "Batch Gradient Descent", "Overfitting", 
                "Underfitting", "Bias-Variance Tradeoff", "Regularization", "L1 Regularization", "L2 Regularization", "Dropout", 
                "Activation Functions", "Sigmoid", "ReLU", "Tanh", "Softmax", "Backpropagation", "Optimization", 
                "Gradient Descent Algorithms", "Learning Rate", "Hyperparameters", "Cross-validation", "Confusion Matrix", 
                "Precision", "Recall", "F1 Score", "Accuracy", "ROC Curve", "AUC-ROC", "Support Vector Machines", 
                "K-Nearest Neighbors", "Random Forest", "Decision Trees", "Naive Bayes", "Logistic Regression", 
                "Linear Regression", "Polynomial Regression", "K-means Clustering", "Hierarchical Clustering", "DBSCAN", 
                "Principal Component Analysis", "Feature Selection", "Dimensionality Reduction", "Data Preprocessing", 
                "Data Cleaning", "Data Normalization", "Data Imputation", "Feature Engineering", "Outlier Detection", 
                "Time Series Analysis", "ARIMA", "Exponential Smoothing", "Seasonality", "Trend", "Forecasting", "Ensemble Methods", 
                "Boosting", "Bagging", "AdaBoost", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost", "Tuning Parameters", 
                "Grid Search", "Random Search", "Hyperparameter Optimization", "Deep Reinforcement Learning", "Q-learning", 
                "SARSA", "Policy Gradient", "Value Iteration", "Monte Carlo Methods", "Markov Decision Processes", "Bellman Equation", 
                "Policy Iteration", "AI Ethics", "AI Fairness", "AI Safety", "AI Governance", "AI Explainability", "Black-box Models", 
                "White-box Models", "Interpretable Machine Learning", "Adversarial Machine Learning", "Data Privacy", "Data Security", 
                "Blockchain", "Edge Computing", "IoT", "Cloud Computing", "Data Lake", "Data Warehouse", "Data Mining", "Web Scraping", 
                "Big Data", "NoSQL", "SQL", "Hadoop", "Spark", "MapReduce", "Hive", "Pig", "Cassandra", "Kafka", "Flume", "Airflow", 
                "ETL", "Batch Processing", "Stream Processing", "SQL Queries", "Join Operations", "Normalization", "Denormalization", 
                "Transaction Management", "ACID Properties", "CAP Theorem", "Sharding", "Database Indexing", "SQL Injection", 
                "Data Integrity", "Data Consistency", "Data Availability", "CAP Theorem", "Event-driven Architecture", 
                "Serverless Computing", "Kubernetes", "Docker", "DevOps", "CI/CD", "Agile", "Scrum", "Kanban", "Test-Driven Development", 
                "Unit Testing", "Integration Testing", "System Testing", "Regression Testing", "Continuous Integration", 
                "Automation Testing", "Load Testing", "Performance Testing", "Monitoring and Logging", "Microservices", "Containerization", 
                "Cloud Services", "AWS", "Google Cloud", "Azure", "GCP", "Cloud Functions", "Lambda", "Serverless Architecture", 
                "Web Development", "HTML", "CSS", "JavaScript", "React", "Vue.js", "Angular", "Node.js", "Express", "Django", "Flask", "API", 
                "RESTful APIs", "GraphQL", "WebSockets", "OAuth", "JWT", "Authentication", "Authorization", "Single Sign-On", "Web Security", 
                "SSL", "TLS", "Cross-Site Scripting", "Cross-Site Request Forgery", "SQL Injection", "Session Management", "Rate Limiting", 
                "Captcha", "Web Performance Optimization", "Content Delivery Network", "API Rate Limiting", "Caching", "Database Sharding", 
                "Load Balancing", "Replication", "Cloud Security", "IAM", "DevSecOps", "Zero Trust Architecture", "Distributed Systems", 
                "Event-driven Systems", "Message Queues", "Pub/Sub", "RabbitMQ", "Kafka", "Microservices Architecture", "API Gateway", 
                "Service Mesh", "Data Consistency", "Fault Tolerance", "Capacitance", "Elasticity", "Vertical Scaling", "Horizontal Scaling", 
                "Auto-scaling", "Load Balancer", "Data Replication", "Partitioning", "Consistency Models", "Eventual Consistency", 
                "Strong Consistency", "CAP Theorem", "Turing Machines", "Computational Complexity", "P vs NP", "Tractability", "Undecidability", 
                "Church-Turing Thesis", "Boolean Algebra", "Logic Gates", "Finite Automata", "Pushdown Automata", "Turing Completeness", 
                "Parallel Computing", "Concurrency", "Distributed Computing", "MapReduce", "Cloud Storage", "IoT Sensors", "IoT Security", 
                "Edge AI", "Fog Computing", "Sensor Networks", "5G", "Autonomous Vehicles", "AI in Robotics", "AI in Healthcare", 
                "AI in Finance", "AI in Cybersecurity", "AI in Education", "AI in Marketing", "AI in Manufacturing", "AI in Supply Chain", 
                "AI in Agriculture", "AI in Retail", "AI in Sports", "AI in Customer Service", "Natural Language Generation", "Text-to-Speech", "Speech-to-Text", "Named Entity Recognition", "Text Classification", "Machine Translation", "Word Embeddings", "GloVe", "Word2Vec", "BERT", "GPT", "Transformer Models", "Attention Mechanism", "Seq2Seq Models", "Pre-trained Models", "Fine-tuning Models", "Transfer Learning", "Few-shot Learning", "One-shot Learning", "Zero-shot Learning", "Neural Machine Translation", "Data Augmentation", "Bias in AI", "AI for Social Good", "Fairness in AI", "Ethical AI", "AI for Sustainability", "Explainable AI", "Interpretability", "AI Accountability", "Human-in-the-loop", "Reinforcement Learning Algorithms", "Monte Carlo Tree Search", "Deep Q-Network", "Actor-Critic", "Policy Optimization", "Markov Chains", "Hidden Markov Models", "Recurrent Neural Networks", "Long Short-Term Memory", "Gated Recurrent Units", "Attention-based Models", "BERT", "GPT-3", "T5", "XLNet", "Transformers", "Graph Neural Networks", "Quantum Computing", "Quantum Algorithms", "Quantum Machine Learning", "Quantum Cryptography", "Blockchain and AI", "Smart Contracts", "AI Governance", "AI Regulation", "AI in Law", "AI in Art", "AI in Music", "AI in Cinema", "AI in Design", "Neural Architecture Search", "Genetic Algorithms", "Swarm Intelligence", "Optimization Algorithms", "Simulated Annealing", "Ant Colony Optimization", "Particle Swarm Optimization", "Evolutionary Algorithms", "Metaheuristics", "Game Theory", "Nash Equilibrium", "Zero-sum Games", "Cooperative Games", "Non-cooperative Games", "Multi-agent Systems", "Autonomous Systems", "Decision Making", "Strategic Decision Making", "Risk Assessment", "Cost-benefit Analysis", "Prediction Models", "Forecasting Models", "Anomaly Detection", "Fraud Detection", "Classification Algorithms", "Regression Algorithms", "Clustering Algorithms", "Dimensionality Reduction", "Feature Engineering", "Feature Scaling", "Data Visualization", "Data Storytelling", "Business Intelligence", "Business Analytics", "AI in Business", "AI in Marketing", "AI in HR", "AI in Operations", "AI for Customer Service", "AI in Sales", "AI for Financial Services", "AI for Healthcare", "AI for Transportation", "AI for Supply Chain Management", "Robotic Process Automation", "AI in Healthcare Diagnostics", "AI in Drug Discovery", "AI in Personalization", "AI Chatbots", "Virtual Assistants", "Conversational AI", "AI-Powered Search", "AI in Smart Homes", "AI in Retail", "Predictive Maintenance", "Image Recognition", "Object Detection", "Speech Recognition", "Natural Language Processing", "Computer Vision", "AI for Education", "AI for Games", "AI for Security", "AI in Agriculture", "AI for Disaster Response", "AI for Climate Change", "Ethics in AI", "AI for Social Justice", "AI for Human Rights" ]

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

def is_study_related(user_input):
    user_words = user_input.lower().split()
    return any(keyword in user_words for keyword in STUDY_KEYWORDS)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    if is_study_related(user_input):
        response = chat_session.send_message(user_input)
        return jsonify({"response": response.text})
    else:
        return jsonify({"response": "I only answer study-related queries. Please ask about exams, topics, or preparation."})

if __name__ == '__main__':
    app.run(debug=True)
