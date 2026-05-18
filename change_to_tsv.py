import csv

input_file = "/home/pintokf/Projects/Microbium/Mouses/mouses_2_data/metadata_all_samples_new.txt"
output_file = "/home/pintokf/Projects/Microbium/Mouses/mouses_2_data/metadata_all_samples_new.tsv"

with open(input_file, "r", newline="", encoding="utf-8") as infile, \
     open(output_file, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile, delimiter="\t")

    for row in reader:
        writer.writerow(row)

# Remove double quotes from each row in the output TSV file
with open(output_file, "r", newline="", encoding="utf-8") as tsvfile:
    rows = tsvfile.readlines()

with open(output_file, "w", newline="", encoding="utf-8") as tsvfile:
    for row in rows:
        tsvfile.write(row.replace('"', ''))

print("Conversion complete!")