import os

# Get the directory where the script is located
folder_path = os.path.dirname(os.path.abspath(__file__))

# Output file name
output_file = os.path.join(folder_path, "pdf_list.txt")

# Collect all PDF filenames
pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

# Write to txt file
with open(output_file, "w", encoding="utf-8") as f:
    for pdf in pdf_files:
        f.write(pdf + "\n")

print(f"Saved {len(pdf_files)} PDF filenames to {output_file}")