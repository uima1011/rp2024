import matplotlib.pyplot as plt
import pandas as pd

# Daten aus der CSV-Datei laden
with open('./data/score.csv') as f:
    lines = f.readlines()

# Werte in eine Liste konvertieren
x = [float(line.strip()) for line in lines]

# In ein Pandas DataFrame konvertieren
x_series = pd.Series(x)

# Rollenden Durchschnitt berechnen
rolling_mean = x_series.rolling(window=100).mean()

# Plotten
plt.figure(figsize=(10, 6))
plt.plot(rolling_mean, color='lightblue', label='Rollender Durchschnitt')
plt.scatter(range(len(x)), x, color='darkblue', s=10, label='Originalwerte')
plt.title("Erfüllungsgrad der Aufgabe")
plt.xlabel("Iteration")
plt.ylabel("Score")
plt.legend()
plt.show()
