# Define the output file path
output_file = "legal_case_reports_qrels.tsv"

# Generate and write the rows
with open(output_file, "w", encoding="utf-8") as f:
    for i in range(770):  # From 0 to 1999
        f.write(f"{i}\t0\t{i}\t1\n")

print(f"Qrels file saved as {output_file}")
