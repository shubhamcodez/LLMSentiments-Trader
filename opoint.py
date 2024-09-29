import json
import codecs

def get_column_names(file_path):
    columns = set()
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with codecs.open(file_path, 'r', encoding=encoding) as file:
                for line in file:
                    try:
                        article = json.loads(line)
                        columns.update(article.keys())
                        break  # We only need to read one article successfully
                    except json.JSONDecodeError:
                        continue  # Skip lines that can't be parsed as JSON
                
                print(f"Successfully read the file using {encoding} encoding.")
                return sorted(columns)
        except UnicodeDecodeError:
            print(f"Failed to read with {encoding} encoding. Trying next...")
    
    print("Failed to read the file with all attempted encodings.")
    return []

if __name__ == "__main__":
    file_path = "english_data.jsonl"
    columns = get_column_names(file_path)
    
    if columns:
        print("\nColumns found in the dataset:")
        for column in columns:
            print(f"- {column}")
    else:
        print("No columns were found. Please check the file and its location.")