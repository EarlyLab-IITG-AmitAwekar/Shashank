import pandas as pd

def get_middle_data_rowwise(csv_file_path):
    """
    Get the middle 10 data points and display them row-wise without any processing
    Save the output only to a text file
    """
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Total rows in CSV: {len(df)}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Get the middle 10 rows
    total_rows = len(df)
    if total_rows < 10:
        print(f"CSV has only {total_rows} rows. Showing all available rows.")
        middle_df = df
    else:
        start_idx = (total_rows - 10) // 2
        end_idx = start_idx + 10
        middle_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        print(f"Showing rows {start_idx + 1} to {end_idx} from original CSV")
    
    # Prepare output for both console and file
    output_lines = []
    output_lines.append("="*100)
    output_lines.append("MIDDLE 10 ROWS - ROW-WISE DATA")
    output_lines.append("="*100)
    
    print("\n" + "="*100)
    print("MIDDLE 10 ROWS - ROW-WISE DATA")
    print("="*100)
    
    # Display each row with all its column data
    for idx, row in middle_df.iterrows():
        row_header = f"\n--- ROW {idx + 1} ---"
        print(row_header)
        output_lines.append(row_header)
        
        for column in middle_df.columns:
            line = f"{column}: {row[column]}"
            print(line)
            output_lines.append(line)
        
        separator = "-" * 100
        print(separator)
        output_lines.append(separator)
    
    # Save to text file only
    output_filename = 'middle_10_rows_analysis.txt'
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        print(f"\nOutput saved to '{output_filename}'")
    except Exception as e:
        print(f"Error saving to text file: {e}")
    
    return middle_df

# Usage
if __name__ == "__main__":
    # Replace 'your_file.csv' with the actual path to your CSV file
    csv_file_path = './llama_phi_strategyqa_final_results.csv'  # UPDATE THIS PATH
    
    # Get middle data row-wise
    middle_data = get_middle_data_rowwise(csv_file_path)
