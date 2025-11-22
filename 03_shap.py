# %% Imports
"""
Task 2-4: SHAP Explainer Creation and Visualization
- Task 2: Create and improve SHAP explainer (TreeExplainer)
- Task 3: Local explanation visualization (Force Plot)
- Task 4: Global explanation visualization (Summary Plot and Bar Plot)
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
import shap

# Create result directory
os.makedirs('../result', exist_ok=True)

# %% Load and preprocess data
print("=" * 60)
print("Task 2: Data Loading and Preprocessing")
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

# %% Task 2: Fit blackbox model and evaluate
print("\n" + "=" * 60)
print("Task 2: Train Random Forest Model")
print("=" * 60)
rf = RandomForestClassifier(n_estimators=100, random_state=2021, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Model performance evaluation
print("\nModel Performance Metrics:")
print(f"F1 Score (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Stroke', 'Stroke']))

# %% Task 2: Create SHAP explainer
print("\n" + "=" * 60)
print("Task 2: Create SHAP Explainer (TreeExplainer)")
print("=" * 60)
explainer = shap.TreeExplainer(rf)
print("SHAP explainer created successfully!")

# Calculate SHAP values for all test samples (for global analysis)
print("\nCalculating SHAP values for all test samples...")
shap_values_all = explainer.shap_values(X_test)

# Handle different return formats from TreeExplainer
if isinstance(shap_values_all, list):
    # List format: [array_class0, array_class1]
    print(f"SHAP values shape: {len(shap_values_all)} classes, each with shape {shap_values_all[0].shape}")
    shap_values_class1_all = shap_values_all[1]  # Use class 1 (Stroke)
else:
    # Array format: (n_samples, n_features, n_classes)
    print(f"SHAP values shape: {shap_values_all.shape}")
    shap_values_class1_all = shap_values_all[:, :, 1]  # Extract class 1 (Stroke)

# For binary classification, we focus on SHAP values for class 1 (Stroke)

# Calculate expected value
expected_value = explainer.expected_value
if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1:
    expected_value_class1 = expected_value[1]
else:
    expected_value_class1 = expected_value
print(f"Expected value (class 1): {expected_value_class1:.4f}")

# %% Task 3: Local explanation visualization (Force Plot)
print("\n" + "=" * 60)
print("Task 3: Local Explanation Visualization (Force Plot)")
print("=" * 60)

# Select several samples for local explanation
sample_indices = [0, 1, 2, 10, 20]
X_samples = X_test.iloc[sample_indices]
y_samples = y_test.iloc[sample_indices]
shap_values_samples_raw = explainer.shap_values(X_samples)

# Handle different return formats
if isinstance(shap_values_samples_raw, list):
    shap_values_samples_class1 = shap_values_samples_raw[1]
else:
    shap_values_samples_class1 = shap_values_samples_raw[:, :, 1]

print(f"\nCreating local explanations for {len(sample_indices)} samples...")

# Force Plot for individual samples
for idx, sample_idx in enumerate(sample_indices):
    print(f"\nSample {sample_idx}:")
    prediction = rf.predict(X_test.iloc[sample_idx:sample_idx+1])[0]
    actual = y_test.iloc[sample_idx]
    print(f"  Actual label: {actual} ({'Stroke' if actual == 1 else 'No Stroke'})")
    print(f"  Predicted label: {prediction} ({'Stroke' if prediction == 1 else 'No Stroke'})")
    
    # Create force plot (for class 1, i.e., Stroke)
    shap.force_plot(
        expected_value_class1,
        shap_values_samples_class1[idx],
        X_samples.iloc[idx:idx+1],
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f'../result/force_plot_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: result/force_plot_sample_{sample_idx}.png")

# Force Plot for multiple samples (HTML format)
print("\nCreating Force Plot for multiple samples (HTML format)...")
shap.initjs()
force_plot_html = shap.force_plot(
    expected_value_class1,
    shap_values_samples_class1,
    X_samples,
    show=False
)
shap.save_html('../result/force_plot_multiple_samples.html', force_plot_html)
print("Saved: result/force_plot_multiple_samples.html")

# %% Task 4: Global explanation visualization
print("\n" + "=" * 60)
print("Task 4: Global Explanation Visualization")
print("=" * 60)

# For computational efficiency, use a subset of test set for global analysis
# If test set is too large, we can sample
n_samples_for_global = min(100, len(X_test))
if len(X_test) > n_samples_for_global:
    print(f"\nUsing {n_samples_for_global} samples from test set for global analysis...")
    sample_indices_global = np.random.choice(len(X_test), n_samples_for_global, replace=False)
    X_test_global = X_test.iloc[sample_indices_global]
    shap_values_global_raw = explainer.shap_values(X_test_global)
    # Handle different return formats
    if isinstance(shap_values_global_raw, list):
        shap_values_global = shap_values_global_raw[1]
    else:
        shap_values_global = shap_values_global_raw[:, :, 1]
else:
    X_test_global = X_test
    shap_values_global = shap_values_class1_all

# 4.1 Summary Plot (dot plot)
print("\n4.1 Creating Summary Plot (dot plot)...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_global, X_test_global, show=False)
plt.tight_layout()
plt.savefig('../result/summary_plot_dot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: result/summary_plot_dot.png")

# 4.2 Summary Plot (bar plot)
print("\n4.2 Creating Summary Plot (bar plot)...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_global, X_test_global, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('../result/summary_plot_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: result/summary_plot_bar.png")

# 4.3 Create bar plot using shap.plots.bar (new API)
try:
    print("\n4.3 Creating feature importance bar plot using shap.plots.bar...")
    shap_values_obj = shap.TreeExplainer(rf).shap_values(X_test_global)
    if isinstance(shap_values_obj, list):
        shap_values_obj = shap_values_obj[1]  # Use SHAP values for class 1
    
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values_obj, show=False)
    plt.tight_layout()
    plt.savefig('../result/bar_plot_features.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: result/bar_plot_features.png")
except Exception as e:
    print(f"Error creating bar plot with new API: {e}")
    print("Using traditional method...")

# 4.4 Feature importance bar plot (manually calculate mean SHAP values)
print("\n4.4 Creating feature importance bar plot (mean SHAP values)...")
mean_shap_values = np.abs(shap_values_global).mean(0)
feature_importance = pd.Series(mean_shap_values, index=X_test_global.columns)
feature_importance = feature_importance.sort_values(ascending=False)

plt.figure(figsize=(12, 8))
feature_importance.plot(kind='barh')
plt.xlabel('Mean |SHAP Value|', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Feature Importance (Based on Mean SHAP Values)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../result/feature_importance_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: result/feature_importance_bar.png")

# Print top 10 most important features
print("\nTop 10 Most Important Features:")
for i, (feature, importance) in enumerate(feature_importance.head(10).items(), 1):
    print(f"  {i}. {feature}: {importance:.4f}")

# 4.5 Waterfall Plot (detailed explanation for a single sample)
print("\n4.5 Creating Waterfall Plot (detailed explanation for a single sample)...")
sample_idx = 0
explanation = shap.Explanation(
    values=shap_values_samples_class1[0],
    base_values=expected_value_class1,
    data=X_samples.iloc[0].values,
    feature_names=X_samples.columns.tolist()
)
plt.figure(figsize=(10, 6))
shap.plots.waterfall(explanation, show=False)
plt.tight_layout()
plt.savefig('../result/waterfall_plot_sample_0.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: result/waterfall_plot_sample_0.png")

# %% Summary
print("\n" + "=" * 60)
print("Task 2-4 Completion Summary")
print("=" * 60)
print("\nCreated visualization files:")
print("  - result/force_plot_sample_*.png (local explanations)")
print("  - result/force_plot_multiple_samples.html (interactive explanations for multiple samples)")
print("  - result/summary_plot_dot.png (global feature importance dot plot)")
print("  - result/summary_plot_bar.png (global feature importance bar plot)")
print("  - result/bar_plot_features.png (feature importance bar plot)")
print("  - result/feature_importance_bar.png (mean SHAP values bar plot)")
print("  - result/waterfall_plot_sample_0.png (detailed explanation for a single sample)")
print("\nAll tasks completed!")

# %%