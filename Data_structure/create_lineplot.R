# טעינת ספריות
library(tidyverse)

# נתיבים - וודא שאלו הנתיבים הנכונים אצלך
fb_file <- "/home/pintokf/Projects/Microbium/Mouses/Data_structure/fb_ratio.tsv"
meta_file <- "/home/pintokf/Projects/Microbium/Mouses/Data_structure/to_catagorial_metadata/metadata_categorical.tsv"
output_plot <- "/home/pintokf/Projects/Microbium/Mouses/Data_structure/FB_ratio_lineplot_restored.png"

# קריאת הנתונים 
fb_data <- read_tsv(fb_file)
# קריאת המטא-דאטה, מדלגים על שורת ההגדרות של QIIME2 אם קיימת
meta_data <- read_tsv(meta_file, comment = "#q2:") 

# תיקון שם העמודה הראשונה כדי שיתאים בשני הקבצים (לפעמים הסולמית עושה בעיות ב-R)
colnames(fb_data)[1] <- "SampleID"
colnames(meta_data)[1] <- "SampleID"

# מיזוג שתי הטבלאות
df <- inner_join(fb_data, meta_data, by = "SampleID")

# חישוב הממוצע ושגיאת התקן (SE) לכל חודש
summary_df <- df %>%
  group_by(AgeMonth) %>%
  summarise(
    N = n(),
    Mean_FB = mean(FB_ratio, na.rm = TRUE),
    SD_FB = sd(FB_ratio, na.rm = TRUE),
    SE_FB = SD_FB / sqrt(N) # חישוב שגיאת התקן
  ) %>%
  # חילוץ המספר של החודש נטו (למשל מתוך "Month_02" נוציא "2")
  mutate(
    Numeric_Age = as.numeric(str_extract(AgeMonth, "\\d+")),
    # בניית התווית המדויקת לציר ה-X, למשל: "2_months\n(N=30)"
    X_Label = paste0(Numeric_Age, "_months\n(N=", N, ")")
  ) %>%
  arrange(Numeric_Age)

# הגדרת סדר התוויות בציר ה-X לפי הגיל כדי שלא יסתדרו אלפביתית
summary_df$X_Label <- factor(summary_df$X_Label, levels = summary_df$X_Label)

# יצירת הגרף
p <- ggplot(summary_df, aes(x = X_Label, y = Mean_FB, group = 1)) +
  # הוספת קווי השגיאה
  geom_errorbar(aes(ymin = Mean_FB - SE_FB, ymax = Mean_FB + SE_FB), 
                width = 0.1, color = "#2c7bb6", linewidth = 0.5) +
  # קו מחבר ונקודות
  geom_line(color = "#2c7bb6", linewidth = 0.5) +
  geom_point(color = "#2c7bb6", size = 1) +
  # כותרות
  labs(title = "Firmicutes/Bacteroidetes",
       x = "Age (Months)",
       y = "Firmicutes/Bacteroidetes") +
  # עיצוב נקי כמו בתמונה
  theme_classic() +
  theme(
    plot.title = element_text(hjust = 1, size = 22, face = "bold"), # כותרת גדולה ומיושרת לימין
    axis.text.x = element_text(size = 9, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title = element_text(size = 12)
  )

# שמירת הפלוט
ggsave(output_plot, p, width = 10, height = 6, dpi = 300)

print(paste("הגרף נשמר בהצלחה בנתיב:", output_plot))