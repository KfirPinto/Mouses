import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

# === נתיבים (התאם לפי הצורך) ===
base_path = "/home/pintokf/Projects/Microbium/Mouses"
z_train_path = f"{base_path}/Locate_model/locate_Z_train_level_6.csv"
z_test_path = f"{base_path}/Locate_model/locate_Z_test_level_6.csv"
metabo_path = f"{base_path}/preprocess_metabolits/preprocessed_metabolites_normalized_z_score.csv"
output_dir = f"{base_path}/Locate_model/Evaluation"

def evaluate_z():
    print("--- 1. Loading Data ---")
    # טעינת ה-Z
    df_z_train = pd.read_csv(z_train_path)
    df_z_test = pd.read_csv(z_test_path)
    
    # טעינת המטבוליטים (האמת)
    df_metabo = pd.read_csv(metabo_path)
    
    # טיפול בשמות עמודות ID
    if 'SampleID' in df_metabo.columns: df_metabo.rename(columns={'SampleID': 'ID'}, inplace=True)
    if 'ID' in df_metabo.columns: df_metabo.set_index('ID', inplace=True)
    else: df_metabo.set_index(df_metabo.columns[0], inplace=True)
    
    # המרת אינדקסים ל-str ליתר ביטחון
    df_metabo.index = df_metabo.index.astype(str)
    df_z_train['ID'] = df_z_train['ID'].astype(str)
    df_z_test['ID'] = df_z_test['ID'].astype(str)

    print(f"Z Train: {df_z_train.shape}, Z Test: {df_z_test.shape}")

    # === 2. Alignment (התאמת שורות) ===
    # אנחנו צריכים את המטבוליטים המתאימים לכל שורה ב-Z
    
    # פונקציית עזר לסינון ומיון
    def align_y_to_z(z_df, y_df):
        # השארת רק דגימות שקיימות ב-Z
        common_ids = [ids for ids in z_df['ID'] if ids in y_df.index]
        
        # סינון ה-Z וה-Y
        z_aligned = z_df[z_df['ID'].isin(common_ids)].set_index('ID')
        y_aligned = y_df.loc[common_ids]
        
        # וידוא שהם באותו סדר בדיוק!
        y_aligned = y_aligned.reindex(z_aligned.index)
        
        return z_aligned, y_aligned

    # הכנת הנתונים לאימון ומבחן
    X_train, Y_train = align_y_to_z(df_z_train, df_metabo)
    X_test, Y_test = align_y_to_z(df_z_test, df_metabo)
    
    print(f"Aligned Train: {X_train.shape}, Aligned Test: {X_test.shape}")

    # === 3. Training a Decoder (Linear Regression) ===
    print("\n--- Training Decoder (Z -> Metabolites) ---")
    # אנחנו לומדים קשר לינארי: Metabolites = Z * W + b
    # אם Z הוא טוב, הקשר הזה אמור להיות חזק
    decoder = LinearRegression()
    decoder.fit(X_train, Y_train)
    
    # === 4. Prediction & Evaluation ===
    print("--- Predicting on Test Set ---")
    Y_pred_matrix = decoder.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred_matrix, index=Y_test.index, columns=Y_test.columns)

    # חישוב קורלציה (Spearman) לכל מטבוליט
    print("--- Calculating Correlations ---")
    corrs = []
    valid_metabolites = 0
    
    for col in Y_test.columns:
        true_vals = Y_test[col]
        pred_vals = Y_pred[col]
        
        # חישוב רק אם יש שונות (לא הכל אפסים)
        if np.std(true_vals) > 0 and np.std(pred_vals) > 0:
            c, _ = spearmanr(true_vals, pred_vals)
            if not np.isnan(c):
                corrs.append(c)
                valid_metabolites += 1
        else:
            corrs.append(0) # במקרה של חוסר שונות

    mean_corr = np.mean(corrs)
    median_corr = np.median(corrs)
    
    print("\n" + "="*40)
    print(f"RESULTS FOR Z QUALITY (Test Set)")
    print("="*40)
    print(f"Mean Spearman Correlation:   {mean_corr:.4f}")
    print(f"Median Spearman Correlation: {median_corr:.4f}")
    print(f"Positive Correlations:       {sum(c > 0 for c in corrs)} / {len(corrs)}")
    print("="*40)
    
    # === 5. Plotting ===
    plt.figure(figsize=(10, 6))
    sns.histplot(corrs, bins=30, kde=True, color="purple")
    plt.axvline(mean_corr, color='red', linestyle='--', label=f'Mean: {mean_corr:.3f}')
    plt.title("Distribution of Reconstruction Accuracy (Z -> Metabolites)")
    plt.xlabel("Spearman Correlation")
    plt.ylabel("Count of Metabolites")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plot_path = f"{output_dir}/z_quality_evaluation_6.png"
    plt.savefig(plot_path)
    print(f"\n✅ Plot saved to: {plot_path}")

if __name__ == "__main__":
    evaluate_z()