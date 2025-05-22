def remove_duplicates(input_file, output_file):
    seen = set()  # To store unique (column 1, column 3) pairs
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            parts = line.split()
            if len(parts) >= 4:  # Ensure there are enough columns
                col1 = parts[0]
                col3 = parts[2]
                if (col1, col3) not in seen:
                    seen.add((col1, col3))
                    outfile.write(line)

# Usage example
remove_duplicates('tevatron/run_passage_emb.10k_aggregated_mapped.txt', 'tevatron/run_passage_emb.10k_aggregated_mapped_dedup.txt')
