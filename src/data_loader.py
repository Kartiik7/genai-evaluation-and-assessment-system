import pandas as pd
import os

def load_mohler(file_path="data/raw/mohler_dataset_edited.csv"):
    """
    Loads the Mohler dataset from the given path, renames columns to a standard format,
    and returns a cleaned pandas DataFrame.
    """
    df = pd.read_csv(file_path)
    
    # Map original columns to the requested standard columns
    column_mapping = {
        'id': 'question_id',
        'question': 'question',
        'desired_answer': 'reference_answer',
        'student_answer': 'student_answer',
        'score_avg': 'human_score'
    }
    
    # Select and rename columns
    df = df.rename(columns=column_mapping)
    df = df[['question_id', 'question', 'reference_answer', 'student_answer', 'human_score']]
    
    return df

if __name__ == "__main__":
    df = load_mohler()
    
    # Save the clean table
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "mohler_clean.csv")
    df.to_csv(out_path, index=False)
    
    # Print requested statistics
    print(f"Number of rows: {len(df)}")
    print(f"Number of unique questions: {df['question_id'].nunique()}")
    print("Missing values per column:")
    print(df.isnull().sum())
    print(f"Number of duplicate rows: {df.duplicated().sum()}")
