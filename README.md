# 🚀 Hate Speech Detection using Machine Learning

## 📌 Abstract
This project focuses on detecting hate speech in text using **Natural Language Processing (NLP)** and **Machine Learning**. A Support Vector Machine (LinearSVC) model trained on a balanced dataset is used along with **TF-IDF vectorization** to classify text as **Hate Speech (1)** or **Non-Hate Speech (0)**.  
A simple **Flask API** with a clean **web interface** is provided for real-time predictions.

---

## ⚙️ Tech Stack
- **Programming Language:** Python  
- **Frameworks:** Flask, Scikit-learn  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Joblib  
- **Frontend:** HTML, CSS, JavaScript  
- **Dataset:** Cleaned & balanced hate-speech dataset  

## 🧑‍💻 How to Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/Hate-Speech-Detection.git
cd Hate-Speech-Detection/src
pip install -r requirements.txt
python train.py
python app.py
```
Visit http://127.0.0.1:5000/ in your browser.

## Results
-**Accuracy:** 82%

-**Confusion Matrix:** 

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/4a6aa043-08c1-499c-a428-1f652fde6beb" />

-**Precision, Recall, F1 Score :** 

<img width="635" height="301" alt="image" src="https://github.com/user-attachments/assets/ece6be6c-6cea-4158-937f-2d2aecb3f2c2" />

## Screenchots

<img width="1919" height="1009" alt="Screenshot 2025-09-13 053850" src="https://github.com/user-attachments/assets/0d6ed99f-c1cd-4cc7-985c-adf65c855ea0" />

<img width="1919" height="1001" alt="Screenshot 2025-09-13 053912" src="https://github.com/user-attachments/assets/3eeae69b-04ab-481b-b3c2-0a8b143755b1" />


## Future Work
-**Train on a larger dataset for better generalization.**   
-**Use Deep Learning (LSTMs / BERT) for improved accuracy.**   
-**Deploy on Render / Vercel / PythonAnywhere for live demo.**   

## Author

**Sneha Gupta**   
-**Contact:** Snehagupta061204@gmail.com
