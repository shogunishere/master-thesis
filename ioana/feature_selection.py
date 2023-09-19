import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from andraz import settings

class FeatureSelection:
    def __init__(self, path):
        self.data = pd.read_pickle(path)
        self.data = self.data.drop(["max_saturation"], axis="columns")
        self.data = self.data.drop(["min_saturation"], axis="columns")
        self.data = self.data.drop(["index"], axis="columns")

    def select_features_by_threshold(self, threshold):
        corr_matrix = self.data.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=True, yticklabels=True)
        plt.title("Correlation Matrix Heatmap")
        plt.show()

        features_to_discard = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] >= threshold:
                    feature_i = corr_matrix.columns[i]
                    feature_j = corr_matrix.columns[j]

                    if feature_i not in features_to_discard:
                        features_to_discard.add(feature_j)

        selected_features_df = self.data.drop(columns=features_to_discard)

        return selected_features_df

if __name__ == "__main__":
    path = Path(settings.PROJECT_DIR) / "ioana/train_features.pickle"
    cfs = FeatureSelection(path)
    threshold = 0.75
    selected_features_df = cfs.select_features_by_threshold(threshold)
    print(f"Selected Features {selected_features_df.columns.tolist()}")


