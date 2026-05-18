# טעינת ספריות
library(tidyverse)
library(Maaslin2)

# --- הגדרת נתיבים ---
feat_file <- "/home/pintokf/Projects/Microbium/Mouses/Data_structure/exported_all_samples/level-6.csv"
meta_file <- "/home/pintokf/Projects/Microbium/Mouses/mouse_data_new/metadata_all_samples_new.tsv"
output_dir <- "/home/pintokf/Projects/Microbium/Mouses/Data_structure/Maaslin2_Results"
output_plot <- paste0(output_dir, "/Akkermansia_lineplot_restored.png")

dir.create(output_dir, showWarnings = FALSE)

# --- קריאת הנתונים ---
raw_features <- read.csv(feat_file, row.names=1, check.names=FALSE)
meta <- read.table(meta_file, sep="\t", header=TRUE, row.names=1, comment.char="")

# ניקוי המטא-דאטה
if(rownames(meta)[1] == "#q2:types") { meta <- meta[-1, ] }
meta$AgeNumeric <- as.numeric(as.character(meta$AgeMonth))

# בידוד רק של העמודות הנומריות (החיידקים) לצורך חישובים
features <- raw_features[, sapply(raw_features, is.numeric)]

# ניקוי שמות החיידקים
clean_names <- colnames(features)
clean_names <- gsub(".*g__", "", clean_names)
clean_names <- gsub("\\[|\\]", "", clean_names)
colnames(features) <- clean_names

# --- 1. הרצת אנליזת MaAsLin2 (רק אם הקבצים עוד לא קיימים) ---
if (!file.exists(paste0(output_dir, "/significant_results.tsv"))) {
  cat("Running MaAsLin2...\n")
  Maaslin2(
    input_data = features,
    input_metadata = meta,
    output = output_dir,
    fixed_effects = c("AgeNumeric"),
    standardize = FALSE,
    plot_scatter = FALSE,
    plot_heatmap = FALSE
  )
} else {
  cat("MaAsLin2 results already exist, skipping to plotting...\n")
}

# --- 2. יצירת הפלוט של Akkermansia ---
cat("Generating Akkermansia plot...\n")

# חישוב שפע יחסי
rel_features <- features / rowSums(features)

# איחוד עם המטא-דאטה
akk_column <- grep("Akkermansia", colnames(rel_features), value = TRUE)[1]

if (is.na(akk_column)) {
    stop("Could not find Akkermansia in the features!")
}

akk_data <- rel_features %>%
  select(all_of(akk_column)) %>%
  rename(Akkermansia = 1) %>%
  rownames_to_column("SampleID") %>%
  inner_join(meta %>% rownames_to_column("SampleID"), by="SampleID")

# חישוב ממוצעים ושגיאות תקן
summary_akk <- akk_data %>%
  filter(!is.na(AgeNumeric)) %>%
  group_by(AgeNumeric) %>%
  summarise(
    N = n(),
    Mean_Abundance = mean(Akkermansia, na.rm=TRUE),
    SE_Abundance = sd(Akkermansia, na.rm=TRUE) / sqrt(N)
  )

# ציור הגרף
p <- ggplot(summary_akk, aes(x = AgeNumeric, y = Mean_Abundance)) +
  geom_errorbar(aes(ymin = Mean_Abundance - SE_Abundance, ymax = Mean_Abundance + SE_Abundance), 
                width = 0.2, color = "#2c7bb6", linewidth = 0.5) +
  geom_line(color = "#2c7bb6", linewidth = 0.5) +
  geom_point(color = "#2c7bb6", size = 1.5) +
  scale_x_continuous(breaks = seq(2, 18, by = 2)) +
  labs(title = "Relative abundance of Akkermansia\nmuciniphila",
       x = "Age (Months)",
       y = "Relative abundance (%)") +
  theme_classic() +
  theme(
    plot.title = element_text(hjust = 1, size = 18, face = "bold"),
    axis.text = element_text(color = "black", size = 10),
    axis.title = element_text(size = 12)
  )

# שמירה
ggsave(output_plot, p, width = 8, height = 5, dpi = 300)
cat(paste("Done! Plot saved to:\n", output_plot, "\n"))