import pickle
import numpy as np
from colorama import Fore, init

# Load Model
with open("model/spam.pkl", "rb") as file:
    model = pickle.load(file)

text = [input("Input SMS Text : ")]
y_pred = model.predict(text)
y_proba = model.predict_proba(text)

init()
if y_pred[0] == 1:
    print(Fore.RED + f"Text diatas merupakan SPAM. Dengan Probabilitas : {np.max(y_proba)}")
else:
    print(Fore.GREEN + f"Text diatas BUKAN SPAM. Dengan Probabilitas : {np.min(y_proba)}")
