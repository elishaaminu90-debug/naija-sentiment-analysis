"""
Run this ONCE at the start of every Colab session, before any other step.
Mounts Google Drive and creates a persistent project folder so your
CSVs, weights, and models survive session restarts / GPU quota resets.
"""

from google.colab import drive
import os

drive.mount('/content/drive')

PROJECT_DIR = "/content/drive/MyDrive/sentiment-project"
os.makedirs(PROJECT_DIR, exist_ok=True)
os.chdir(PROJECT_DIR)

print(f"Working directory set to: {PROJECT_DIR}")
print("All files saved from here on (CSVs, weights, models) will persist across sessions.")
print("\nIMPORTANT: Add this cell to the TOP of every notebook for this project,")
print("run it first, and it must finish (Drive auth) before running Steps 1-4.")
