import os
import pandas as pd

def combine_csv_texts(folder_path, text_column="section,text"):
    all_texts = []
    try:        
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".csv"):
                file_path = os.path.join(folder_path, file_name)
                print(f"Reading {file_path}")
                df = pd.read_csv(file_path, encoding='utf-8')  
                print(f"First few rows of {file_path}:\n", df.head())  
                
                if text_column in df.columns:
                    all_texts += df[text_column].dropna().astype(str).tolist()
                else:
                    print(f"Column '{text_column}' not found in {file_path}")
        
    except Exception as e:
        print(f"Error reading files: {e}")
    
    return "\n\n".join(all_texts)


train_folder = "C:/Users/ronal\Mini LLM/section-stories/train"
val_folder = "C:/Users/ronal\Mini LLM/section-stories/val"


train_text = combine_csv_texts(train_folder, text_column="text")
val_text = combine_csv_texts(val_folder, text_column="text")


if not train_text:
    print("No text data found for training.")
else:
    with open("train.txt", "w", encoding="utf-8") as f:
        f.write(train_text)

if not val_text:
    print("No text data found for validation.")
else:
    with open("val.txt", "w", encoding="utf-8") as f:
        f.write(val_text)

print("✅ Done writing train.txt and val.txt")
