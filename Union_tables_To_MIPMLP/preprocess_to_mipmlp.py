import pandas as pd
from pathlib import Path

base_path = Path("mouses_data/clean_fastq/exports")

otu_path = base_path / "otu.csv"
taxonomy_path = base_path / "tax.tsv/taxonomy.csv"
preprocess_path = "Union_tables_To_MIPMLP/for_preprocess.csv"

otu = pd.read_csv(otu_path, index_col=0)
tax = pd.read_csv(taxonomy_path, index_col=0)
tax = tax.loc[otu.index]

# Add taxonomy to OTU and transpose
otu["taxonomy"] = tax["Taxon"]

otu_t = otu.T
otu_t.index.name = 'ID'

otu_t.to_csv(preprocess_path)