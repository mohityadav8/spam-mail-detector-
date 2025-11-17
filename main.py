import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

# ---------------------------
# 1. Load Dataset
# ---------------------------

mail_data = pd.read_csv("mail_data.csv", encoding="latin-1")
mail_data = mail_data.where(pd.notnull(mail_data), "")

mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham',  'Category'] = 1

X = mail_data['Message']
Y = mail_data['Category'].astype(int)

# ---------------------------
# 2. Train/Test Split
# ---------------------------

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=3
)

# ---------------------------
# 3. TF-IDF Vectorizer
# ---------------------------

vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# ---------------------------
# 4. Train Model
# ---------------------------

model = LogisticRegression()
model.fit(X_train_features, Y_train)

# ---------------------------
# 5. GUI Application
# ---------------------------

def check_spam():
    user_input = text_box.get("1.0", tk.END).strip()
    if user_input == "":
        messagebox.showwarning("Warning", "Please enter a message!")
        return

    input_features = vectorizer.transform([user_input])
    prediction = model.predict(input_features)

    if prediction[0] == 1:
        result = "Not Spam ✔"
    else:
        result = "Spam ❌"

    messagebox.showinfo("Result", f"Message is: {result}")


# Tkinter Window
root = tk.Tk()
root.title("Spam Mail Detector")
root.geometry("400x300")

label = tk.Label(root, text="Enter your message:", font=("Arial", 12))
label.pack(pady=10)

text_box = tk.Text(root, height=8, width=40)
text_box.pack()

check_button = tk.Button(root, text="Check Spam", font=("Arial", 12), command=check_spam)
check_button.pack(pady=15)

root.mainloop()
