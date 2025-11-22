# %% Imports
"""
Task 6: Explain SVM Model using Kernel SHAP
- Use SVM as black box model (different from Random Forest in Task 2)
- Implement Kernel SHAP explainer
- Visualize results and compare with Task 2 TreeExplainer results
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import DataLoader
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, classification_report
import shap
import warnings
warnings.filterwarnings('ignore')

# Create result directory
os.makedirs('../result', exist_ok=True)

# %% Load and preprocess data
print("=" * 60)
print("Task 6: Data Loading and Preprocessing")
print("=" * 60)
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()

# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Oversample the train data to handle class imbalance
X_train, y_train = data_loader.oversample(X_train, y_train)
print(f"Training set shape after oversampling: {X_train.shape}")

# %% Train SVM model
print("\n" + "=" * 60)
print("Task 6: Train SVM Model")
print("=" * 60)

# Use RBF kernel SVM, set probability=True to enable predict_proba
svm = SVC(kernel='rbf', probability=True, random_state=2021, C=1.0, gamma='scale')
print("Starting SVM model training...")
svm.fit(X_train, y_train)
print("Training completed!")

# Evaluate model performance
y_pred = svm.predict(X_test)
print("\nSVM Model Performance Metrics:")
print(f"F1 Score (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Stroke', 'Stroke']))

# %% Create Kernel SHAP explainer
print("\n" + "=" * 60)
print("Task 6: Create Kernel SHAP Explainer")
print("=" * 60)

# Kernel SHAP requires a background dataset to calculate expected values
# For computational efficiency, we sample a smaller background dataset from training set
background_size = min(100, len(X_train))
print(f"Sampling {background_size} samples from training set as background dataset...")
background_indices = np.random.choice(len(X_train), background_size, replace=False)
X_background = X_train.iloc[background_indices]

# Define prediction function (required by Kernel SHAP)
def predict_proba_wrapper(X):
    """Wrapper for SVM's predict_proba method to make it compatible with SHAP"""
    if isinstance(X, pd.DataFrame):
        return svm.predict_proba(X)
    else:
        return svm.predict_proba(pd.DataFrame(X, columns=X_train.columns))

# Create Kernel SHAP explainer
print("Creating Kernel SHAP explainer...")
explainer = shap.KernelExplainer(predict_proba_wrapper, X_background)
print("Kernel SHAP explainer created successfully!")

# %% Calculate SHAP values
print("\n" + "=" * 60)
print("Task 6: Calculate SHAP Values")
print("=" * 60)

# Kernel SHAP computation is slow, so we only calculate for a subset of test set
n_samples = min(50, len(X_test))
print(f"Calculating SHAP values for {n_samples} samples from test set (this may take some time)...")

# Randomly select samples
sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
X_test_samples = X_test.iloc[sample_indices]

# Calculate SHAP values (only for class 1, i.e., Stroke)
print("Calculating SHAP values, please wait...")
shap_values = explainer.shap_values(X_test_samples, nsamples=100)  # nsamples controls sampling amount

# For binary classification, shap_values may be a list or array
if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]  # SHAP values for class 1 (Stroke)
elif len(shap_values.shape) == 3:
    # 3D array: (n_samples, n_features, n_classes)
    shap_values_class1 = shap_values  # Keep 3D for now, will extract class 1 later
else:
    shap_values_class1 = shap_values

print(f"SHAP values calculation completed! Shape: {shap_values_class1.shape}")

# Get expected value
expected_value = explainer.expected_value
if isinstance(expected_value, (list, np.ndarray)):
    expected_value_class1 = expected_value[1] if len(expected_value) > 1 else expected_value[0]
else:
    expected_value_class1 = expected_value
print(f"Expected value (class 1): {expected_value_class1:.4f}")

# %% Visualize SHAP values
print("\n" + "=" * 60)
print("Task 6: Visualize SHAP Values")
print("=" * 60)

# 6.1 Summary Plot (dot plot)
print("\n6.1 Creating Summary Plot (dot plot)...")
# Extract class 1 SHAP values if 3D
if len(shap_values_class1.shape) == 3:
    shap_values_for_summary = shap_values_class1[:, :, 1]
else:
    shap_values_for_summary = shap_values_class1

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_for_summary, X_test_samples, show=False)
plt.tight_layout()
plt.savefig('../result/svm_shap_summary_plot_dot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: result/svm_shap_summary_plot_dot.png")

# 6.2 Summary Plot (bar plot)
print("\n6.2 Creating Summary Plot (bar plot)...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_for_summary, X_test_samples, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('../result/svm_shap_summary_plot_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: result/svm_shap_summary_plot_bar.png")

# 6.3 Feature importance bar plot (mean SHAP values)
print("\n6.3 Creating feature importance bar plot (mean SHAP values)...")
# Handle case where shap_values_class1 might be 3D (n_samples, n_features, n_classes)
if len(shap_values_class1.shape) == 3:
    shap_values_class1 = shap_values_class1[:, :, 1]  # Extract class 1
mean_shap_values = np.abs(shap_values_class1).mean(0)
feature_importance = pd.Series(mean_shap_values, index=X_test_samples.columns)
feature_importance = feature_importance.sort_values(ascending=False)

plt.figure(figsize=(12, 8))
feature_importance.plot(kind='barh')
plt.xlabel('Mean |SHAP Value|', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('SVM Model Feature Importance (Based on Kernel SHAP)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../result/svm_feature_importance_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: result/svm_feature_importance_bar.png")

# Print top 10 most important features
print("\nTop 10 Most Important Features for SVM Model:")
for i, (feature, importance) in enumerate(feature_importance.head(10).items(), 1):
    print(f"  {i}. {feature}: {importance:.4f}")

# 6.4 Force Plot (single sample)
print("\n6.4 Creating Force Plot (single sample)...")
sample_idx = 0
prediction = svm.predict(X_test_samples.iloc[sample_idx:sample_idx+1])[0]
print(f"Sample {sample_idx} prediction: {prediction} ({'Stroke' if prediction == 1 else 'No Stroke'})")

# Ensure shap_values_class1 is 2D for force plot
if len(shap_values_class1.shape) == 3:
    shap_values_for_plot = shap_values_class1[sample_idx, :, 1]
else:
    shap_values_for_plot = shap_values_class1[sample_idx]

shap.force_plot(
    expected_value_class1,
    shap_values_for_plot,
    X_test_samples.iloc[sample_idx:sample_idx+1],
    matplotlib=True,
    show=False
)
plt.tight_layout()
plt.savefig('../result/svm_force_plot_sample_0.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: result/svm_force_plot_sample_0.png")

# 6.5 Waterfall Plot
print("\n6.5 Creating Waterfall Plot...")
# Ensure shap_values_class1 is 2D for waterfall plot
if len(shap_values_class1.shape) == 3:
    shap_values_for_waterfall = shap_values_class1[0, :, 1]
else:
    shap_values_for_waterfall = shap_values_class1[0]

explanation = shap.Explanation(
    values=shap_values_for_waterfall,
    base_values=expected_value_class1,
    data=X_test_samples.iloc[0].values,
    feature_names=X_test_samples.columns.tolist()
)
plt.figure(figsize=(10, 6))
shap.plots.waterfall(explanation, show=False)
plt.tight_layout()
plt.savefig('../result/svm_waterfall_plot_sample_0.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: result/svm_waterfall_plot_sample_0.png")

# %% Comparison with Task 2 (Random Forest)
print("\n" + "=" * 60)
print("Task 6: Comparison with Task 2 (Random Forest)")
print("=" * 60)

# Load Task 2 results (if available)
print("\nModel Comparison:")
print(f"{'Metric':<20} {'SVM':<15} {'Random Forest (Task 2)'}")
print("-" * 60)
print(f"{'Accuracy':<20} {accuracy_score(y_test, y_pred):.4f} {'(Run Task 2 to view)'}")
print(f"{'F1 Score (macro)':<20} {f1_score(y_test, y_pred, average='macro'):.4f} {'(Run Task 2 to view)'}")

print("\nExplainer Comparison:")
print(f"{'Explainer Type':<20} {'Kernel SHAP':<15} {'TreeExplainer'}")
print(f"{'Applicable Models':<20} {'Any Model':<15} {'Tree Models'}")
print(f"{'Computation Speed':<20} {'Slower':<15} {'Fast'}")
print(f"{'Accuracy':<20} {'Accurate':<15} {'Accurate'}")

print("\n" + "=" * 60)
print("Task 6 Completion Summary")
print("=" * 60)
print("\nCreated visualization files:")
print("  - result/svm_shap_summary_plot_dot.png (global feature importance dot plot)")
print("  - result/svm_shap_summary_plot_bar.png (global feature importance bar plot)")
print("  - result/svm_feature_importance_bar.png (mean SHAP values bar plot)")
print("  - result/svm_force_plot_sample_0.png (local explanation)")
print("  - result/svm_waterfall_plot_sample_0.png (detailed explanation for a single sample)")
print("\nTask 6 completed!")

# %%