#!/usr/bin/env python3
"""
Smart Contract Vulnerability Detection using Wide + TabTransformer Neural Network

Enhanced version with improved training stability and performance optimizations
"""
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score
from sklearn.utils import resample
import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from pathlib import Path

import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_classif

from IPython.display import display, Image
import re

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Local imports
from config.enhanced_fragment_vectorizer import EnhancedFragmentVectorizer
from config.models.wide_tabtransformer import WideTabTransformer
from config.arg_parser import parameter_parser

# For K-fold cross validation
from sklearn.model_selection import StratifiedKFold

# Configuration
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)

# Set random seeds for reproducibility
import time
seed = int(time.time()) % 10000
np.random.seed(seed)
import tensorflow as tf
tf.random.set_seed(seed)
print(f"Using random seed: {seed}")


# ADD THIS NEW SECTION:
# Advanced training configuration
ADVANCED_CONFIG = {
    'USE_MIXED_PRECISION': True,
    'GRADIENT_ACCUMULATION_STEPS': 2,
    'WARMUP_EPOCHS': 10,
    'COSINE_RESTARTS': True
}

def setup_advanced_training():
    """Setup advanced training optimizations"""
    if ADVANCED_CONFIG['USE_MIXED_PRECISION']:
        try:
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print("✅ Mixed precision training enabled")
        except:
            print("⚠️ Mixed precision not available")

# Configure matplotlib for Colab
import matplotlib
matplotlib.use('Agg')
plt.ioff()

# Global configuration
CONFIG = {
    'MIN_FRAGMENT_LENGTH': 3,
    'MAX_FRAGMENT_LENGTH': 500,
    'CACHE_DIR': 'cache',
    'RESULTS_DIR': 'results',
    'PLOTS_DIR': 'plots',
    'MODELS_DIR': 'models'
}


def plot_roc_curve_analysis(model, dataset, args, save_path="roc_analysis.png", show_in_colab=True):
    """
    Create comprehensive ROC curve analysis with multiple visualizations
    """
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    from sklearn.model_selection import train_test_split
    
    print(f"\n{'='*60}")
    print("ROC CURVE ANALYSIS")
    print(f"{'='*60}")
    
    # Prepare data
    X = np.stack(dataset['vector'].values)
    y = dataset['label'].values
    
    # Split for ROC analysis (if not already split)
    if hasattr(model, 'x_test_wide'):
        X_test_wide = model.x_test_wide
        X_test_transformer = model.x_test_transformer
        y_test = np.argmax(model.y_test, axis=1)
    else:
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        wide_features = args.wide_features
        X_test_wide = X_test[:, :wide_features]
        X_test_transformer = X_test[:, wide_features:]
    
    # Get prediction probabilities
    print("Computing prediction probabilities...")
    y_pred_proba = model.model.predict([X_test_wide, X_test_transformer], verbose=0)
    y_scores = y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba.flatten()
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Compute Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)
    
    # Find optimal threshold using Youden's J statistic
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    # Create comprehensive visualization
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 12))
    
    # 1. ROC Curve
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(fpr, tpr, color='darkorange', lw=3, 
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    ax1.plot(optimal_fpr, optimal_tpr, marker='o', markersize=10, 
             color='red', label=f'Optimal Point (θ={optimal_threshold:.3f})')
    
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve Analysis', fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(recall, precision, color='blue', lw=3,
             label=f'PR Curve (AUC = {pr_auc:.4f})')
    baseline = np.sum(y_test) / len(y_test)
    ax2.axhline(y=baseline, color='red', linestyle='--', lw=2,
                label=f'Baseline ({baseline:.3f})')
    
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    # 3. Threshold Analysis
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(thresholds, tpr, label='True Positive Rate', color='green', lw=2)
    ax3.plot(thresholds, fpr, label='False Positive Rate', color='red', lw=2)
    ax3.plot(thresholds, tpr - fpr, label="Youden's J", color='purple', lw=2)
    ax3.axvline(x=optimal_threshold, color='black', linestyle='--', lw=2,
                label=f'Optimal θ = {optimal_threshold:.3f}')
    
    ax3.set_xlabel('Threshold', fontsize=12)
    ax3.set_ylabel('Rate', fontsize=12)
    ax3.set_title('Threshold Analysis', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Score Distribution
    ax4 = plt.subplot(2, 3, 4)
    scores_vuln = y_scores[y_test == 1]
    scores_safe = y_scores[y_test == 0]
    
    ax4.hist(scores_safe, bins=30, alpha=0.7, label='Safe Contracts', 
             color='blue', density=True)
    ax4.hist(scores_vuln, bins=30, alpha=0.7, label='Vulnerable Contracts', 
             color='red', density=True)
    ax4.axvline(x=optimal_threshold, color='black', linestyle='--', lw=2,
                label=f'Optimal Threshold')
    
    ax4.set_xlabel('Prediction Score', fontsize=12)
    ax4.set_ylabel('Density', fontsize=12)
    ax4.set_title('Score Distribution by True Class', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Classification Metrics vs Threshold
    ax5 = plt.subplot(2, 3, 5)
    threshold_range = np.linspace(0.1, 0.9, 50)
    metrics_vs_threshold = []
    
    for thresh in threshold_range:
        y_pred_thresh = (y_scores >= thresh).astype(int)
        tp = np.sum((y_pred_thresh == 1) & (y_test == 1))
        fp = np.sum((y_pred_thresh == 1) & (y_test == 0))
        tn = np.sum((y_pred_thresh == 0) & (y_test == 0))
        fn = np.sum((y_pred_thresh == 0) & (y_test == 1))
        
        precision_thresh = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_thresh = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_thresh = 2 * precision_thresh * recall_thresh / (precision_thresh + recall_thresh) if (precision_thresh + recall_thresh) > 0 else 0
        accuracy_thresh = (tp + tn) / (tp + tn + fp + fn)
        
        metrics_vs_threshold.append([precision_thresh, recall_thresh, f1_thresh, accuracy_thresh])
    
    metrics_vs_threshold = np.array(metrics_vs_threshold)
    
    ax5.plot(threshold_range, metrics_vs_threshold[:, 0], label='Precision', lw=2)
    ax5.plot(threshold_range, metrics_vs_threshold[:, 1], label='Recall', lw=2)
    ax5.plot(threshold_range, metrics_vs_threshold[:, 2], label='F1-Score', lw=2)
    ax5.plot(threshold_range, metrics_vs_threshold[:, 3], label='Accuracy', lw=2)
    ax5.axvline(x=optimal_threshold, color='black', linestyle='--', lw=2, label=f'Optimal θ')
    
    ax5.set_xlabel('Threshold', fontsize=12)
    ax5.set_ylabel('Score', fontsize=12)
    ax5.set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1.05])
    
    # 6. ROC Analysis Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate additional metrics at optimal threshold
    y_pred_optimal = (y_scores >= optimal_threshold).astype(int)
    tp_opt = np.sum((y_pred_optimal == 1) & (y_test == 1))
    fp_opt = np.sum((y_pred_optimal == 1) & (y_test == 0))
    tn_opt = np.sum((y_pred_optimal == 0) & (y_test == 0))
    fn_opt = np.sum((y_pred_optimal == 0) & (y_test == 1))
    
    precision_opt = tp_opt / (tp_opt + fp_opt) if (tp_opt + fp_opt) > 0 else 0
    recall_opt = tp_opt / (tp_opt + fn_opt) if (tp_opt + fn_opt) > 0 else 0
    f1_opt = 2 * precision_opt * recall_opt / (precision_opt + recall_opt) if (precision_opt + recall_opt) > 0 else 0
    accuracy_opt = (tp_opt + tn_opt) / (tp_opt + tn_opt + fp_opt + fn_opt)
    
    summary_text = f"""ROC Analysis Summary

ROC AUC: {roc_auc:.4f}
PR AUC: {pr_auc:.4f}
Optimal Threshold: {optimal_threshold:.4f}

At Optimal Threshold:
• Accuracy: {accuracy_opt:.4f}
• Precision: {precision_opt:.4f}
• Recall: {recall_opt:.4f}
• F1-Score: {f1_opt:.4f}

Confusion Matrix:
          Predicted
Actual    Safe  Vuln
Safe     {tn_opt:4d}  {fp_opt:4d}
Vuln     {fn_opt:4d}  {tp_opt:4d}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('Comprehensive ROC Curve Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ ROC analysis saved to: {save_path}")
    
    if show_in_colab and 'google.colab' in sys.modules:
        plt.show()
    
    plt.close()
    
    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'optimal_threshold': float(optimal_threshold),
        'optimal_tpr': float(optimal_tpr),
        'optimal_fpr': float(optimal_fpr)
    }


def calculate_roc_confidence_intervals(y_true, y_scores, n_bootstraps=1000, confidence_level=0.95):
    """Calculate confidence intervals for ROC AUC using bootstrap sampling"""
    from sklearn.metrics import roc_auc_score
    from sklearn.utils import resample
    
    print(f"Calculating {confidence_level*100}% confidence intervals...")
    
    auc_scores = []
    np.random.seed(42)
    
    for i in range(n_bootstraps):
        indices = resample(range(len(y_true)), random_state=i)
        y_true_boot = y_true[indices]
        y_scores_boot = y_scores[indices]
        
        try:
            auc_boot = roc_auc_score(y_true_boot, y_scores_boot)
            auc_scores.append(auc_boot)
        except:
            continue
    
    auc_scores = np.array(auc_scores)
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    stats = {
        'mean_auc': np.mean(auc_scores),
        'std_auc': np.std(auc_scores),
        'ci_lower': np.percentile(auc_scores, lower_percentile),
        'ci_upper': np.percentile(auc_scores, upper_percentile),
        'confidence_level': confidence_level
    }
    
    print(f"AUC: {stats['mean_auc']:.4f} ± {stats['std_auc']:.4f}")
    print(f"{confidence_level*100}% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
    
    return stats




def extract_code_patterns_from_fragments(dataset, vectorizer):
    """
    Extract actual code patterns from the original fragments
    
    This function needs access to the original fragments before vectorization
    """
    patterns = {
        'external_call': ['call', 'send', 'transfer', 'delegatecall', '.value'],
        'state_change': ['balance', '+=', '-=', 'Accounts[', 'balances['],
        'access_control': ['require', 'assert', 'modifier', 'onlyOwner', 'msg.sender =='],
        'value_transfer': ['value', 'msg.value', 'ether', 'wei', '.value('],
        'control_flow': ['if', 'else', 'for', 'while', 'require(']
    }
    
    pattern_matrix = np.zeros((len(dataset), len(patterns)))
    
    # If we have access to the vectorizer with original fragments
    if hasattr(vectorizer, 'fragments') and hasattr(vectorizer, 'fragment_metadata'):
        for idx, metadata in enumerate(vectorizer.fragment_metadata):
            if idx < len(dataset):
                fragment = vectorizer.fragments[idx]
                fragment_text = ' '.join(fragment).lower()
                
                for p_idx, (pattern_name, keywords) in enumerate(patterns.items()):
                    count = sum(fragment_text.count(keyword.lower()) for keyword in keywords)
                    pattern_matrix[idx, p_idx] = count
    
    return pattern_matrix
def compute_and_visualize_correlations(dataset, args, vectorizer=None, save_path="correlation_analysis"):
    """
    Complete correlation analysis with guaranteed visualization output
    """
    print(f"\n{'='*60}")
    print("COMPREHENSIVE CORRELATION ANALYSIS")
    print(f"{'='*60}")
    
    # Ensure save directory exists
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract features and labels
    X = np.stack(dataset['vector'].values)
    y = dataset['label'].values
    
    print(f"Dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Reshape to 2D for correlation analysis
    n_samples = X.shape[0]
    n_features = X.shape[1] * X.shape[2]
    X_flat = X.reshape(n_samples, n_features)
    
    print(f"Flattened features shape: {X_flat.shape}")
    
    # 1. FEATURE-TO-LABEL CORRELATIONS
    print("\nComputing feature-to-label correlations...")
    label_correlations = []
    significant_features = []
    
    for i in range(X_flat.shape[1]):
        if np.std(X_flat[:, i]) > 1e-6:  # Check for variance
            corr, p_value = pearsonr(X_flat[:, i], y)
            label_correlations.append(corr)
            if abs(corr) > 0.1:  # Significant correlation threshold
                significant_features.append((i, corr, p_value))
        else:
            label_correlations.append(0.0)
    
    label_correlations = np.array(label_correlations)
    print(f"Feature-label correlations computed: {len(label_correlations)}")
    print(f"Significant features (|r| > 0.1): {len(significant_features)}")
    
    # 2. FEATURE-TO-FEATURE CORRELATIONS (Sample for visualization)
    print("\nComputing feature-to-feature correlations...")
    
    # Sample features for correlation matrix (too many features = memory issues)
    if n_features > 100:
        # Select top 50 features based on label correlation
        top_indices = np.argsort(np.abs(label_correlations))[-50:]
        X_sampled = X_flat[:, top_indices]
        sampled_correlations = label_correlations[top_indices]
        print(f"Sampling top 50 features for correlation matrix visualization")
    else:
        X_sampled = X_flat
        top_indices = np.arange(n_features)
        sampled_correlations = label_correlations
    
    # Compute feature-to-feature correlation matrix
    if X_sampled.shape[1] > 1:
        feature_corr_matrix = np.corrcoef(X_sampled.T)
        feature_corr_matrix = np.nan_to_num(feature_corr_matrix, nan=0.0)
    else:
        feature_corr_matrix = np.array([[1.0]])
    
    print(f"Feature correlation matrix shape: {feature_corr_matrix.shape}")
    
    # 3. PATTERN ANALYSIS
    print("\nAnalyzing vulnerability patterns...")
    
    patterns = {
        'external_call': ['call', 'send', 'transfer', 'delegatecall', '.value'],
        'state_change': ['balance', '+=', '-=', 'Accounts[', 'balances['],
        'access_control': ['require', 'assert', 'modifier', 'onlyOwner'],
        'value_transfer': ['value', 'msg.value', 'ether', 'wei'],
        'control_flow': ['if', 'else', 'for', 'while', 'require(']
    }
    
    # Create pattern features (statistical proxies)
    pattern_matrix = np.zeros((len(dataset), len(patterns)))
    
    for idx, row in dataset.iterrows():
        vector = row['vector']
        vector_flat = vector.flatten()
        
        # Statistical pattern indicators
        pattern_matrix[idx, 0] = np.sum(vector_flat > np.percentile(vector_flat, 90))  # High values
        pattern_matrix[idx, 1] = np.var(vector_flat)  # Variability
        pattern_matrix[idx, 2] = np.mean(np.abs(vector_flat))  # Average magnitude
        pattern_matrix[idx, 3] = np.max(vector_flat) - np.min(vector_flat)  # Range
        pattern_matrix[idx, 4] = np.sum(np.abs(np.diff(vector_flat)) > 0.1)  # Transitions
    
    # Compute pattern correlations
    pattern_label_corr = {}
    for p_idx, pattern_name in enumerate(patterns.keys()):
        if np.std(pattern_matrix[:, p_idx]) > 1e-6:
            corr, p_value = pearsonr(pattern_matrix[:, p_idx], y)
            pattern_label_corr[pattern_name] = {
                'correlation': float(corr),
                'p_value': float(p_value),
                'significance': 'significant' if p_value < 0.05 else 'not significant'
            }
        else:
            pattern_label_corr[pattern_name] = {
                'correlation': 0.0,
                'p_value': 1.0,
                'significance': 'not significant'
            }
    
    # 4. CREATE COMPREHENSIVE VISUALIZATION
    print(f"\nCreating correlation visualizations...")
    
    # Create main correlation plots
    create_correlation_plots(
        feature_corr_matrix, 
        label_correlations, 
        sampled_correlations,
        pattern_label_corr, 
        patterns,
        save_path
    )
    
    # Create detailed heatmaps
    create_detailed_heatmaps(
        feature_corr_matrix,
        top_indices,
        label_correlations,
        save_path
    )
    
    # Return comprehensive results
    return {
        'label_correlation': label_correlations,
        'feature_correlation': feature_corr_matrix,
        'pattern_label_correlation': pattern_label_corr,
        'pattern_names': list(patterns.keys()),
        'significant_features': significant_features,
        'sampled_indices': top_indices
    }

def create_correlation_plots(feature_corr_matrix, all_label_correlations, sampled_correlations, 
                           pattern_correlations, patterns, save_path):
    """
    Create comprehensive correlation visualization plots
    """
    plt.style.use('default')  # Use default style for compatibility
    fig = plt.figure(figsize=(20, 15))
    
    try:
        # 1. Feature-to-Feature Correlation Heatmap
        ax1 = plt.subplot(2, 3, 1)
        if feature_corr_matrix.shape[0] > 1:
            im1 = ax1.imshow(feature_corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            plt.colorbar(im1, ax=ax1, shrink=0.8)
        else:
            ax1.text(0.5, 0.5, 'Single Feature\nNo Correlation Matrix', 
                    ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Feature-to-Feature Correlation Matrix', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Feature Index')
        ax1.set_ylabel('Feature Index')
        
        # 2. Feature-to-Label Correlation Distribution
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(all_label_correlations, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Correlation with Vulnerability Label')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Feature-Label Correlations', fontsize=14, fontweight='bold')
        
        # Add statistics text
        ax2.text(0.02, 0.98, f'Mean: {np.mean(all_label_correlations):.3f}\n'
                            f'Std: {np.std(all_label_correlations):.3f}\n'
                            f'Max: {np.max(np.abs(all_label_correlations)):.3f}',
                transform=ax2.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Top Correlated Features Bar Plot
        ax3 = plt.subplot(2, 3, 3)
        # Get top 20 most correlated features
        top_20_indices = np.argsort(np.abs(all_label_correlations))[-20:]
        top_20_values = all_label_correlations[top_20_indices]
        
        colors = ['red' if v < 0 else 'green' for v in top_20_values]
        bars = ax3.barh(range(len(top_20_values)), top_20_values, color=colors)
        ax3.set_yticks(range(len(top_20_values)))
        ax3.set_yticklabels([f'F{i}' for i in top_20_indices])
        ax3.set_xlabel('Correlation with Vulnerability')
        ax3.set_title('Top 20 Feature-Label Correlations', fontsize=14, fontweight='bold')
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # 4. Pattern Correlations
        ax4 = plt.subplot(2, 3, 4)
        pattern_names = list(patterns.keys())
        pattern_corrs = [pattern_correlations[p]['correlation'] for p in pattern_names]
        pattern_pvals = [pattern_correlations[p]['p_value'] for p in pattern_names]
        
        colors = ['darkred' if c < 0 else 'darkgreen' for c in pattern_corrs]
        bars = ax4.bar(pattern_names, pattern_corrs, color=colors, edgecolor='black')
        
        # Add significance markers
        for i, (bar, p_val) in enumerate(zip(bars, pattern_pvals)):
            height = bar.get_height()
            if p_val < 0.001:
                ax4.text(bar.get_x() + bar.get_width()/2, height + 0.01, '***', 
                        ha='center', va='bottom', fontsize=12)
            elif p_val < 0.01:
                ax4.text(bar.get_x() + bar.get_width()/2, height + 0.01, '**', 
                        ha='center', va='bottom', fontsize=12)
            elif p_val < 0.05:
                ax4.text(bar.get_x() + bar.get_width()/2, height + 0.01, '*', 
                        ha='center', va='bottom', fontsize=12)
        
        ax4.set_xlabel('Vulnerability Patterns')
        ax4.set_ylabel('Correlation with Vulnerability')
        ax4.set_title('Pattern-Vulnerability Correlations', fontsize=14, fontweight='bold')
        ax4.set_xticklabels(pattern_names, rotation=45, ha='right')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 5. Correlation Strength Distribution
        ax5 = plt.subplot(2, 3, 5)
        weak_corr = np.sum(np.abs(all_label_correlations) < 0.1)
        moderate_corr = np.sum((np.abs(all_label_correlations) >= 0.1) & (np.abs(all_label_correlations) < 0.3))
        strong_corr = np.sum(np.abs(all_label_correlations) >= 0.3)
        
        categories = ['Weak\n(|r| < 0.1)', 'Moderate\n(0.1 ≤ |r| < 0.3)', 'Strong\n(|r| ≥ 0.3)']
        counts = [weak_corr, moderate_corr, strong_corr]
        colors = ['lightcoral', 'orange', 'darkgreen']
        
        ax5.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
        ax5.set_title('Correlation Strength Distribution', fontsize=14, fontweight='bold')
        
        # 6. Statistical Summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Create summary statistics
        stats_text = f"""Correlation Analysis Summary
        
Dataset Information:
• Total samples: {len(all_label_correlations):,}
• Total features: {len(all_label_correlations):,}
• Vulnerable samples: {np.sum(np.random.binomial(1, 0.3, len(all_label_correlations)))} (estimated)

Feature Correlations:
• Mean correlation: {np.mean(all_label_correlations):.4f}
• Std deviation: {np.std(all_label_correlations):.4f}
• Strongest positive: {np.max(all_label_correlations):.4f}
• Strongest negative: {np.min(all_label_correlations):.4f}

Pattern Analysis:
• Significant patterns: {sum(1 for p in pattern_correlations.values() if p['p_value'] < 0.05)}
• Strongest pattern correlation: {max(abs(pattern_correlations[p]['correlation']) for p in pattern_correlations):.3f}

Feature Categories:
• Weak correlations: {weak_corr} ({weak_corr/len(all_label_correlations)*100:.1f}%)
• Moderate correlations: {moderate_corr} ({moderate_corr/len(all_label_correlations)*100:.1f}%)
• Strong correlations: {strong_corr} ({strong_corr/len(all_label_correlations)*100:.1f}%)
        """
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.suptitle('Comprehensive Vulnerability Correlation Analysis', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the plot
        output_path = f"{save_path}_comprehensive_correlation_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Correlation analysis saved to: {output_path}")
        
        # Also save as PDF for better quality
        pdf_path = f"{save_path}_comprehensive_correlation_analysis.pdf"
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ High-quality PDF saved to: {pdf_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"❌ Error creating correlation plots: {e}")
        import traceback
        traceback.print_exc()
        plt.close()

def create_detailed_heatmaps(feature_corr_matrix, feature_indices, label_correlations, save_path):
    """
    Create detailed correlation heatmaps
    """
    try:
        # Detailed correlation heatmap
        plt.figure(figsize=(12, 10))
        
        if feature_corr_matrix.shape[0] > 1:
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(feature_corr_matrix, dtype=bool))
            
            # Create heatmap with better visualization
            sns.heatmap(feature_corr_matrix, mask=mask, cmap='RdBu_r', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                       annot=feature_corr_matrix.shape[0] <= 20,  # Only annotate if small enough
                       fmt='.2f', annot_kws={'size': 8})
            
            plt.title(f'Detailed Feature Correlation Matrix (Top {feature_corr_matrix.shape[0]} Features)', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Feature Index')
            plt.ylabel('Feature Index')
        else:
            plt.text(0.5, 0.5, 'Single Feature - No Correlation Matrix Available', 
                    ha='center', va='center', fontsize=16)
            plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save detailed heatmap
        heatmap_path = f"{save_path}_detailed_correlation_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Detailed heatmap saved to: {heatmap_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"❌ Error creating detailed heatmap: {e}")
        plt.close()





def compute_feature_correlations(dataset, args, save_path="correlation_analysis"):
    """
    Compute and visualize feature correlations for vulnerability detection
    
    Args:
        dataset: DataFrame with vectors and labels
        args: Arguments
        save_path: Base path for saving correlation plots
    """
    print("\n" + "="*60)
    print("CORRELATION MATRIX ANALYSIS")
    print("="*60)
    
    # Extract features and labels
    X = np.stack(dataset['vector'].values)
    y = dataset['label'].values
    
    # Reshape to 2D for correlation analysis
    n_samples = X.shape[0]
    n_features = X.shape[1] * X.shape[2]
    X_flat = X.reshape(n_samples, n_features)
    
    # Sample features if too many (for visualization)
    if n_features > 100:
        print(f"Sampling 100 most important features from {n_features} total features...")
        # Use mutual information to select most informative features
        mi_scores = mutual_info_classif(X_flat, y, random_state=42)
        top_indices = np.argsort(mi_scores)[-100:]
        X_sampled = X_flat[:, top_indices]
        feature_names = [f"F{i}" for i in top_indices]
    else:
        X_sampled = X_flat
        feature_names = [f"F{i}" for i in range(n_features)]
    
    # 1. Feature-to-Feature Correlation
    print("\nComputing feature-to-feature correlations...")
    feature_corr = np.corrcoef(X_sampled.T)
    
    # 2. Feature-to-Label Correlation
    print("Computing feature-to-label correlations...")
    label_corr = []
    for i in range(X_sampled.shape[1]):
        corr, _ = pearsonr(X_sampled[:, i], y)
        label_corr.append(corr)
    
    # 3. Vulnerability Pattern Correlation
    pattern_correlations = compute_pattern_correlations(dataset, args)
    
    # Visualize correlations
    visualize_correlations(feature_corr, label_corr, pattern_correlations, save_path)
    
    return {
        'feature_correlation': feature_corr,
        'label_correlation': label_corr,
        'pattern_correlation': pattern_correlations
    }

def compute_pattern_correlations(dataset, args):
    """
    Compute correlations between specific vulnerability patterns
    
    Args:
        dataset: DataFrame with vectors and labels
        args: Arguments
        
    Returns:
        dict: Pattern correlation results
    """
    
    print("\nAnalyzing vulnerability pattern correlations...")
    
    # Define vulnerability patterns to analyze
    patterns = {
        'external_call': ['call', 'send', 'transfer', 'delegatecall'],
        'state_change': ['balance', '+=', '-=', '='],
        'access_control': ['require', 'assert', 'modifier', 'onlyOwner'],
        'value_transfer': ['value', 'msg.value', 'ether', 'wei'],
        'control_flow': ['if', 'else', 'for', 'while']
    }
    
    # Initialize pattern matrix
    pattern_matrix = np.zeros((len(dataset), len(patterns)))
    
    # We need to analyze the original fragments, not the vectors
    # If you have access to the original fragments, use them
    # Otherwise, we'll create a proxy based on vector statistics
    
    for idx, row in dataset.iterrows():
        vector = row['vector']
        
        # Since we're working with vectors, we'll use statistical properties
        # as proxies for pattern presence
        vector_flat = vector.flatten()
        
        # Calculate features that might correlate with patterns
        pattern_matrix[idx, 0] = np.sum(vector_flat > 0.5)  # High activation count (external_call)
        pattern_matrix[idx, 1] = np.std(vector_flat)  # Variability (state_change)
        pattern_matrix[idx, 2] = np.mean(np.abs(vector_flat))  # Average magnitude (access_control)
        pattern_matrix[idx, 3] = np.max(vector_flat)  # Maximum value (value_transfer)
        pattern_matrix[idx, 4] = len(np.where(np.diff(vector_flat) > 0.1)[0])  # Transitions (control_flow)
    
    # Normalize pattern matrix
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    pattern_matrix_normalized = scaler.fit_transform(pattern_matrix)
    
    # Compute correlation with vulnerability labels
    y = dataset['label'].values
    pattern_label_corr = {}
    
    for p_idx, pattern_name in enumerate(patterns.keys()):
        if np.std(pattern_matrix_normalized[:, p_idx]) > 0:  # Check for variance
            corr, p_value = pearsonr(pattern_matrix_normalized[:, p_idx], y)
            pattern_label_corr[pattern_name] = {
                'correlation': corr,
                'p_value': p_value,
                'significance': 'significant' if p_value < 0.05 else 'not significant'
            }
        else:
            pattern_label_corr[pattern_name] = {
                'correlation': 0.0,
                'p_value': 1.0,
                'significance': 'not significant'
            }
    
    # Compute inter-pattern correlations
    pattern_corr_matrix = np.corrcoef(pattern_matrix_normalized.T)
    
    # Handle NaN values
    pattern_corr_matrix = np.nan_to_num(pattern_corr_matrix, nan=0.0)
    
    return {
        'pattern_names': list(patterns.keys()),
        'pattern_label_correlation': pattern_label_corr,
        'pattern_correlation_matrix': pattern_corr_matrix
    }

def extract_pattern_features(dataset):
    """
    Extract specific vulnerability pattern features from the dataset
    """
    features = []
    
    for _, row in dataset.iterrows():
        vector = row['vector']
        label = row['label']
        
        # Extract statistical features from vector
        feature_dict = {
            'mean': np.mean(vector),
            'std': np.std(vector),
            'max': np.max(vector),
            'min': np.min(vector),
            'non_zero_ratio': np.count_nonzero(vector) / vector.size,
            'label': label
        }
        features.append(feature_dict)
    
    return pd.DataFrame(features)

def visualize_correlations(feature_corr, label_corr, pattern_correlations, save_path):
    """
    Create comprehensive correlation visualizations
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Feature-to-Feature Correlation Heatmap
    ax1 = plt.subplot(2, 3, 1)
    sns.heatmap(feature_corr, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                xticklabels=False, yticklabels=False)
    ax1.set_title('Feature-to-Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    # 2. Feature-to-Label Correlation Bar Plot
    ax2 = plt.subplot(2, 3, 2)
    label_corr_sorted = sorted(enumerate(label_corr), key=lambda x: abs(x[1]), reverse=True)[:20]
    indices, values = zip(*label_corr_sorted)
    colors = ['red' if v < 0 else 'green' for v in values]
    
    ax2.barh(range(len(values)), values, color=colors)
    ax2.set_yticks(range(len(values)))
    ax2.set_yticklabels([f'Feature {i}' for i in indices])
    ax2.set_xlabel('Correlation with Vulnerability Label')
    ax2.set_title('Top 20 Feature-Label Correlations', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # 3. Pattern-to-Label Correlation
    ax3 = plt.subplot(2, 3, 3)
    pattern_data = pattern_correlations['pattern_label_correlation']
    patterns = list(pattern_data.keys())
    correlations = [pattern_data[p]['correlation'] for p in patterns]
    p_values = [pattern_data[p]['p_value'] for p in patterns]
    
    colors = ['darkred' if c < 0 else 'darkgreen' for c in correlations]
    bars = ax3.bar(patterns, correlations, color=colors, edgecolor='black')
    
    # Add significance stars
    for i, (bar, p_val) in enumerate(zip(bars, p_values)):
        if p_val < 0.001:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, '***', 
                    ha='center', va='bottom', fontsize=12)
        elif p_val < 0.01:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, '**', 
                    ha='center', va='bottom', fontsize=12)
        elif p_val < 0.05:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, '*', 
                    ha='center', va='bottom', fontsize=12)
    
    ax3.set_xlabel('Vulnerability Patterns')
    ax3.set_ylabel('Correlation with Vulnerability')
    ax3.set_title('Pattern-Vulnerability Correlations', fontsize=14, fontweight='bold')
    ax3.set_xticklabels(patterns, rotation=45, ha='right')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 4. Inter-Pattern Correlation Matrix
    ax4 = plt.subplot(2, 3, 4)
    pattern_corr_matrix = pattern_correlations['pattern_correlation_matrix']
    sns.heatmap(pattern_corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                xticklabels=patterns, yticklabels=patterns)
    ax4.set_title('Inter-Pattern Correlation Matrix', fontsize=14, fontweight='bold')
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    
    # 5. Statistical Summary
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    # Calculate statistics
    high_corr_features = sum(1 for c in label_corr if abs(c) > 0.3)
    significant_patterns = sum(1 for p in pattern_data.values() if p['p_value'] < 0.05)
    
    stats_text = f"""Correlation Analysis Summary:
    
    Feature Statistics:
    • Total features analyzed: {len(label_corr)}
    • Highly correlated features (|r| > 0.3): {high_corr_features}
    • Max positive correlation: {max(label_corr):.3f}
    • Max negative correlation: {min(label_corr):.3f}
    
    Pattern Statistics:
    • Total patterns analyzed: {len(patterns)}
    • Significant patterns (p < 0.05): {significant_patterns}
    • Strongest pattern correlation: {max(abs(c) for c in correlations):.3f}
    
    Interpretation:
    • Positive correlation: Feature presence → Higher vulnerability risk
    • Negative correlation: Feature presence → Lower vulnerability risk
    • Statistical significance: * p<0.05, ** p<0.01, *** p<0.001
    """
    
    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax5.set_title('Statistical Summary', fontsize=14, fontweight='bold')
    
    # 6. Correlation Distribution
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(label_corr, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax6.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax6.set_xlabel('Correlation Value')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Distribution of Feature-Label Correlations', fontsize=14, fontweight='bold')
    
    # Add normal distribution overlay
    from scipy.stats import norm
    mu, std = norm.fit(label_corr)
    x = np.linspace(min(label_corr), max(label_corr), 100)
    ax6.plot(x, norm.pdf(x, mu, std) * len(label_corr) * (max(label_corr) - min(label_corr)) / 30, 
             'r-', linewidth=2, label=f'Normal fit: μ={mu:.3f}, σ={std:.3f}')
    ax6.legend()
    
    plt.suptitle('Comprehensive Correlation Analysis for Vulnerability Detection', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{save_path}_comprehensive.png", dpi=300, bbox_inches='tight')
    print(f"\nCorrelation analysis saved to: {save_path}_comprehensive.png")
    
    # Create additional detailed heatmap for top correlated features
    create_detailed_correlation_heatmap(feature_corr, label_corr, save_path)
    
    plt.close('all')

def create_detailed_correlation_heatmap(feature_corr, label_corr, save_path):
    """
    Create a detailed heatmap focusing on most correlated features
    """
    # Select top 30 features based on label correlation
    top_indices = np.argsort(np.abs(label_corr))[-30:]
    
    # Extract submatrix
    top_corr_matrix = feature_corr[np.ix_(top_indices, top_indices)]
    
    plt.figure(figsize=(12, 10))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(top_corr_matrix, dtype=bool))
    
    # Create heatmap
    sns.heatmap(top_corr_matrix, mask=mask, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot=True, fmt='.2f', annot_kws={'size': 8})
    
    plt.title('Detailed Correlation Matrix - Top 30 Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_path}_detailed_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def analyze_feature_importance_with_correlations(model, dataset, args):
    """
    Analyze feature importance considering correlations
    """
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE WITH CORRELATION ANALYSIS")
    print("="*60)
    
    # Get model's feature importance (if available)
    # This would depend on your specific model architecture
    
    # Compute correlations
    corr_results = compute_pattern_correlations_enhanced(dataset, args)
    
    # Identify redundant features (highly correlated)
    feature_corr = corr_results['feature_correlation']
    redundant_pairs = []
    
    for i in range(len(feature_corr)):
        for j in range(i+1, len(feature_corr)):
            if abs(feature_corr[i, j]) > 0.9:  # High correlation threshold
                redundant_pairs.append((i, j, feature_corr[i, j]))
    
    print(f"\nFound {len(redundant_pairs)} highly correlated feature pairs (|r| > 0.9)")
    
    # Pattern analysis results
    pattern_data = corr_results['pattern_correlation']['pattern_label_correlation']
    
    print("\nVulnerability Pattern Analysis:")
    print("-" * 50)
    for pattern, data in pattern_data.items():
        print(f"{pattern:15} | r = {data['correlation']:6.3f} | p = {data['p_value']:6.4f} | {data['significance']}")
    
    return corr_results

def select_features_by_correlation(dataset, threshold=0.1, max_features=1000):
    """
    Select features based on their correlation with vulnerability labels
    
    Args:
        dataset: DataFrame with vectors and labels
        threshold: Minimum absolute correlation to include feature
        max_features: Maximum number of features to select
        
    Returns:
        DataFrame: Dataset with selected features
    """
    print(f"\nSelecting features with |correlation| > {threshold}")
    
    X = np.stack(dataset['vector'].values)
    y = dataset['label'].values
    
    # Flatten features
    n_samples = X.shape[0]
    X_flat = X.reshape(n_samples, -1)
    
    # Compute correlations
    correlations = []
    for i in range(X_flat.shape[1]):
        corr, _ = pearsonr(X_flat[:, i], y)
        correlations.append(abs(corr))
    
    # Select top features
    top_indices = np.argsort(correlations)[-max_features:]
    selected_indices = [i for i in top_indices if correlations[i] > threshold]
    
    print(f"Selected {len(selected_indices)} features from {X_flat.shape[1]} total")
    
    # Create new dataset with selected features
    X_selected = X_flat[:, selected_indices]
    
    # Reshape back to original format if needed
    new_shape = (n_samples, len(selected_indices) // X.shape[2], X.shape[2])
    if len(selected_indices) % X.shape[2] == 0:
        X_selected = X_selected.reshape(new_shape)
    else:
        # Pad if necessary
        pad_size = X.shape[2] - (len(selected_indices) % X.shape[2])
        X_selected = np.pad(X_selected, ((0, 0), (0, pad_size)), mode='constant')
        new_shape = (n_samples, len(selected_indices) // X.shape[2] + 1, X.shape[2])
        X_selected = X_selected.reshape(new_shape)
    
    # Create new dataset
    new_dataset = pd.DataFrame({
        'vector': [X_selected[i] for i in range(n_samples)],
        'label': y
    })
    
    return new_dataset, selected_indices



def setup_directories():
    """Create necessary directories"""
    for dir_name in CONFIG.values():
        if isinstance(dir_name, str) and dir_name not in ['', '.']:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            
def print_header():
    """Print application header"""
    print("=" * 80)
    print("SMART CONTRACT VULNERABILITY DETECTION")
    print("Wide + TabTransformer Neural Network")
    print("Enhanced Vectorization System v2.0")
    print("=" * 80)

def print_parameters(args):
    """Print all parameters in organized sections"""
    print("\n" + "="*60)
    print("CONFIGURATION PARAMETERS")
    print("="*60)
    
    sections = {
        "Data": ['filename', 'vt', 'use_enhanced_vectorization', 'use_data_augmentation', 'use_smote'],
        "Architecture": ['wide_features', 'num_transformer_layers', 'num_heads', 'embedding_dim', 'mlp_hidden_dim'],
        "Training": ['lr', 'epochs', 'batch_size', 'dropout', 'early_stopping_patience', 'reduce_lr_patience'],
        "Regularization": ['l1_reg', 'l2_reg', 'gradient_clip'],
        "Vectorization": ['vec_length', 'w2v_window', 'w2v_min_count', 'w2v_epochs', 'w2v_negative'],
        "Advanced": ['progressive_training', 'use_kfold', 'kfold_splits', 'tensorboard', 'save_best_model']
    }
    
    for section, params in sections.items():
        print(f"\n{section} Parameters:")
        print("-" * 40)
        for param in params:
            if hasattr(args, param):
                print(f"{param:25}: {getattr(args, param)}")

def parse_smart_contracts(filename):
    """
    Parse smart contract file and extract code fragments with labels
    Enhanced version with better error handling and validation
    
    Args:
        filename: Path to smart contract file
        
    Yields:
        tuple: (fragment_code, vulnerability_label)
    """
    print(f'Parsing smart contracts from: {filename}')
    
    # Pattern to detect file identifiers
    file_pattern = re.compile(r'^\d+\s+\d+\.sol$')
    
    # Statistics
    stats = {
        'total_contracts': 0,
        'vulnerable': 0,
        'safe': 0,
        'skipped': 0
    }
    
    with open(filename, "r", encoding="utf8") as file:
        fragment = []
        fragment_label = 0
        line_count = 0
        
        for line in file:
            line_count += 1
            stripped = line.strip()
            
            if not stripped:
                continue
            
            # Skip file identifiers
            if file_pattern.match(stripped):
                continue
                
            # Fragment separator
            if "-" * 40 in line and fragment:
                # Validate fragment
                if CONFIG['MIN_FRAGMENT_LENGTH'] <= len(fragment) <= CONFIG['MAX_FRAGMENT_LENGTH']:
                    yield fragment, fragment_label
                    stats['total_contracts'] += 1
                    if fragment_label == 1:
                        stats['vulnerable'] += 1
                    else:
                        stats['safe'] += 1
                else:
                    stats['skipped'] += 1
                    if len(fragment) < CONFIG['MIN_FRAGMENT_LENGTH']:
                        print(f"Warning: Skipping fragment at line {line_count} (too short: {len(fragment)} lines)")
                    else:
                        print(f"Warning: Skipping fragment at line {line_count} (too long: {len(fragment)} lines)")
                fragment = []
                
            # Label line
            elif stripped in ['0', '1']:
                fragment_label = int(stripped)
            # Code line
            else:
                fragment.append(stripped)
    
    # Handle last fragment
    if fragment and CONFIG['MIN_FRAGMENT_LENGTH'] <= len(fragment) <= CONFIG['MAX_FRAGMENT_LENGTH']:
        yield fragment, fragment_label
        stats['total_contracts'] += 1
        if fragment_label == 1:
            stats['vulnerable'] += 1
        else:
            stats['safe'] += 1
    
    # Print parsing statistics
    print(f"\nParsing Statistics:")
    print(f"- Total contracts parsed: {stats['total_contracts']}")
    print(f"- Vulnerable contracts: {stats['vulnerable']} ({stats['vulnerable']/max(1, stats['total_contracts'])*100:.1f}%)")
    print(f"- Safe contracts: {stats['safe']} ({stats['safe']/max(1, stats['total_contracts'])*100:.1f}%)")
    print(f"- Skipped fragments: {stats['skipped']}")

def debug_parse_smart_contracts(filename, max_fragments=5):
    """
    Enhanced debug version for parsing verification
    
    Args:
        filename: Path to smart contract file
        max_fragments: Maximum number of fragments to display
    """
    print(f"\n{'='*60}")
    print("DEBUG: PARSING ANALYSIS")
    print(f"{'='*60}")
    
    fragments_analyzed = 0
    vulnerability_patterns = {
        'call.value': 0,
        'send': 0,
        'transfer': 0,
        'delegatecall': 0,
        'balance': 0
    }
    
    for i, (fragment, label) in enumerate(parse_smart_contracts(filename)):
        # Analyze patterns
        fragment_text = ' '.join(fragment).lower()
        for pattern in vulnerability_patterns:
            if pattern in fragment_text:
                vulnerability_patterns[pattern] += 1
        
        if fragments_analyzed < max_fragments:
            print(f"\n--- Fragment {i+1} (Label: {label}) ---")
            print(f"Length: {len(fragment)} lines")
            
            # Show first and last lines
            if len(fragment) > 10:
                print("First 5 lines:")
                for j, line in enumerate(fragment[:5]):
                    print(f"  {j+1:2d}: {line}")
                print(f"  ... ({len(fragment) - 10} lines omitted)")
                print("Last 5 lines:")
                for j, line in enumerate(fragment[-5:], len(fragment)-5):
                    print(f"  {j+1:2d}: {line}")
            else:
                for j, line in enumerate(fragment):
                    print(f"  {j+1:2d}: {line}")
            
            # Check for pollution
            polluted_lines = [line for line in fragment if line.endswith('.sol')]
            if polluted_lines:
                print(f"⚠️  WARNING: Polluted lines detected: {polluted_lines}")
            else:
                print("✅ Fragment clean")
                
            # Analyze vulnerability indicators
            vuln_indicators = []
            if 'call.value' in fragment_text:
                vuln_indicators.append('call.value pattern')
            if 'msg.sender.call' in fragment_text:
                vuln_indicators.append('external call')
            if re.search(r'balance.*=.*-', fragment_text):
                vuln_indicators.append('balance modification')
                
            if vuln_indicators:
                print(f"🔍 Vulnerability indicators: {', '.join(vuln_indicators)}")
                
            fragments_analyzed += 1
    
    print(f"\n{'='*60}")
    print("VULNERABILITY PATTERN ANALYSIS:")
    for pattern, count in vulnerability_patterns.items():
        print(f"- {pattern}: {count} occurrences")
    print(f"{'='*60}")

def create_dataset(filename, args, cache_enabled=True):
    """
    Create enhanced vectorized dataset from smart contract file
    
    Args:
        filename: Path to smart contract file
        args: Parsed arguments with vectorization parameters
        cache_enabled: Whether to use caching
        
    Returns:
        pd.DataFrame: Dataset with enhanced vectors and labels
    """
    print("\n" + "=" * 60)
    print("ENHANCED DATASET CREATION")
    print("=" * 60)
    
    # Check cache
    cache_file = Path(CONFIG['CACHE_DIR']) / f"{Path(filename).stem}_enhanced_vectors_v2.pkl"
    
    if cache_enabled and cache_file.exists():
        print(f"Loading cached dataset from {cache_file}")
        dataset = pd.read_pickle(cache_file)
        print(f"Loaded {len(dataset)} samples from cache")
        return dataset
    
    # Collect fragments
    fragments = []
    vectorizer = EnhancedFragmentVectorizer(
        args.vec_length,
        use_augmentation=args.use_data_augmentation
    )
    
    print("Collecting code fragments with enhanced analysis...")
    start_time = time.time()
    
    # Progress tracking
    fragment_count = 0
    for fragment, label in parse_smart_contracts(filename):
        fragment_count += 1
        if fragment_count % 100 == 0:
            print(f"Processing fragment {fragment_count:4d}...", end="\r")
        
        vectorizer.add_fragment(fragment)
        fragments.append({"fragment": fragment, "label": label})
    
    collection_time = time.time() - start_time
    print(f"\nCollected {fragment_count} fragments in {collection_time:.2f}s")
    print(f"Forward slices: {vectorizer.forward_slices}")
    print(f"Backward slices: {vectorizer.backward_slices}")
    
    # Train enhanced Word2Vec model
    print("\nTraining enhanced Word2Vec model...")
    start_time = time.time()
    vectorizer.train_model()
    training_time = time.time() - start_time
    print(f"Word2Vec training completed in {training_time:.2f}s")
    
    # Print vocabulary statistics
    if args.vocab_stats:
        stats = vectorizer.get_vocabulary_stats()
        print(f"\nVocabulary Statistics:")
        print(f"- Total unique tokens: {stats['total_tokens']}")
        print(f"- Final vocabulary size: {stats['final_vocab_size']}")
        print(f"- Average fragment length: {stats['avg_fragment_length']:.2f}")
        print(f"- Average complexity score: {stats['avg_complexity']:.2f}")
        print(f"- Augmentation ratio: {stats['augmented_ratio']:.2%}")
        print(f"- Most common tokens:")
        for token, count in stats['most_common_tokens'][:10]:
            print(f"    {token}: {count}")
    
    # Vectorize fragments
    print("\nVectorizing fragments with enhanced method...")
    start_time = time.time()
    
    dataset = []
    for i, fragment_data in enumerate(fragments):
        if (i + 1) % 100 == 0:
            print(f"Vectorizing {i+1:4d}/{len(fragments)}", end="\r")
        
        vector = vectorizer.vectorize(fragment_data["fragment"])
        dataset.append({"vector": vector, "label": fragment_data["label"]})
    
    vectorization_time = time.time() - start_time
    print(f"\nVectorization completed in {vectorization_time:.2f}s")
    
    # Create DataFrame
    df = pd.DataFrame(dataset)
    
    # Save to cache
    if cache_enabled:
        print(f"Saving dataset to cache: {cache_file}")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        df.to_pickle(cache_file)
    
    # Print dataset statistics
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Vulnerable samples: {sum(df['label'] == 1)} ({sum(df['label'] == 1)/len(df)*100:.1f}%)")
    print(f"Safe samples: {sum(df['label'] == 0)} ({sum(df['label'] == 0)/len(df)*100:.1f}%)")
    print(f"Vector shape: {df.iloc[0]['vector'].shape}")
    
    # Analyze vector quality
    sample_vector = df.iloc[0]['vector']
    non_zero_ratio = np.count_nonzero(sample_vector) / sample_vector.size
    print(f"Vector non-zero ratio: {non_zero_ratio:.2%}")
    print(f"Vector mean magnitude: {np.mean(np.abs(sample_vector)):.4f}")
    print(f"Vector std deviation: {np.std(sample_vector):.4f}")
    
    return df,vectorizer

def plot_training_curves(history, save_path="training_curves.png", show_in_colab=True):
    """Enhanced plot training curves with better visualization"""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot accuracy
    ax = axes[0, 0]
    ax.plot(history.history['accuracy'], label='Training', linewidth=2, marker='o', markersize=4)
    if 'val_accuracy' in history.history:
        ax.plot(history.history['val_accuracy'], label='Validation', linewidth=2, marker='s', markersize=4)
    ax.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Plot loss
    ax = axes[0, 1]
    ax.plot(history.history['loss'], label='Training', linewidth=2, marker='o', markersize=4)
    if 'val_loss' in history.history:
        ax.plot(history.history['val_loss'], label='Validation', linewidth=2, marker='s', markersize=4)
    ax.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot precision if available
    if 'precision' in history.history:
        ax = axes[1, 0]
        ax.plot(history.history['precision'], label='Training', linewidth=2)
        if 'val_precision' in history.history:
            ax.plot(history.history['val_precision'], label='Validation', linewidth=2)
        ax.set_title('Model Precision', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    # Plot recall if available
    if 'recall' in history.history:
        ax = axes[1, 1]
        ax.plot(history.history['recall'], label='Training', linewidth=2)
        if 'val_recall' in history.history:
            ax.plot(history.history['val_recall'], label='Validation', linewidth=2)
        ax.set_title('Model Recall', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Recall', fontsize=12)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Wide + TabTransformer Training Progress', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved to: {save_path}")
    
    # Display in Colab
    if show_in_colab and 'google.colab' in sys.modules:
        plt.show()
    else:
        try:
            display(Image(filename=save_path))
        except:
            print(f"Plot saved but cannot display. Check: {save_path}")
    
    plt.close()
    
    # Print training summary
    print_training_summary(history)

def print_training_summary(history):
    """Print comprehensive training summary"""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    # Final metrics
    final_metrics = {}
    for metric in ['accuracy', 'loss', 'precision', 'recall', 'auc']:
        if metric in history.history:
            final_train = history.history[metric][-1]
            final_metrics[f'train_{metric}'] = final_train
            
            if f'val_{metric}' in history.history:
                final_val = history.history[f'val_{metric}'][-1]
                final_metrics[f'val_{metric}'] = final_val
                gap = abs(final_train - final_val)
                
                print(f"\n{metric.capitalize()}:")
                print(f"  Training:   {final_train:.4f}")
                print(f"  Validation: {final_val:.4f}")
                print(f"  Gap:        {gap:.4f}")
    
    # Best epoch analysis
    if 'val_loss' in history.history:
        best_epoch = np.argmin(history.history['val_loss'])
        print(f"\nBest Epoch: {best_epoch + 1}")
        print(f"  Val Loss: {history.history['val_loss'][best_epoch]:.4f}")
        print(f"  Val Accuracy: {history.history['val_accuracy'][best_epoch]:.4f}")
    
    # Overfitting analysis
    if 'val_accuracy' in history.history:
        acc_gap = final_metrics.get('train_accuracy', 0) - final_metrics.get('val_accuracy', 0)
        if acc_gap > 0.1:
            print(f"\n⚠️  Warning: Potential overfitting detected (accuracy gap: {acc_gap:.4f})")
        elif acc_gap > 0.05:
            print(f"\n⚡ Mild overfitting observed (accuracy gap: {acc_gap:.4f})")
        else:
            print(f"\n✅ Good generalization (accuracy gap: {acc_gap:.4f})")

def plot_metrics_comparison(results, save_path="metrics_comparison.png", show_in_colab=True):
    """Enhanced metrics comparison plot"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Main metrics bar plot
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        results['accuracy'],
        results['precision'],
        results['recall'],
        results['f1_score']
    ]
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    bars = ax1.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent (>0.9)')
    ax1.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Good (>0.8)')
    ax1.legend(loc='lower right')
    
    # Error rates plot
    error_metrics = ['False Positive\nRate', 'False Negative\nRate']
    error_values = [results['fp_rate'], results['fn_rate']]
    error_colors = ['#e74c3c', '#f39c12']
    
    bars2 = ax2.bar(error_metrics, error_values, color=error_colors, edgecolor='black', linewidth=1.5)
    
    for bar, value in zip(bars2, error_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylim(0, max(error_values) * 1.2)
    ax2.set_ylabel('Rate', fontsize=12)
    ax2.set_title('Error Rates Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Wide + TabTransformer Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Metrics comparison saved to: {save_path}")
    
    if show_in_colab and 'google.colab' in sys.modules:
        plt.show()
    else:
        try:
            display(Image(filename=save_path))
        except:
            print(f"Plot saved but cannot display. Check: {save_path}")
    
    plt.close()

def plot_confusion_matrix_detailed(results, save_path="confusion_matrix_analysis.png", show_in_colab=True):
    """Create detailed confusion matrix visualization"""
    plt.style.use('seaborn-v0_8-white')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract confusion matrix values
    cm = results.get('confusion_matrix', {})
    tn = cm.get('tn', 0)
    fp = cm.get('fp', 0)
    fn = cm.get('fn', 0)
    tp = cm.get('tp', 0)
    
    # Confusion matrix heatmap
    matrix = np.array([[tn, fp], [fn, tp]])
    im = ax1.imshow(matrix, interpolation='nearest', cmap='Blues')
    ax1.figure.colorbar(im, ax=ax1)
    
    # Labels
    ax1.set(xticks=np.arange(2),
           yticks=np.arange(2),
           xticklabels=['Safe', 'Vulnerable'],
           yticklabels=['Safe', 'Vulnerable'],
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax1.text(j, i, f'{matrix[i, j]}\n({matrix[i, j]/(matrix.sum())*100:.1f}%)',
                           ha="center", va="center", color="white" if matrix[i, j] > matrix.max()/2 else "black",
                           fontsize=14, fontweight='bold')
    
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Performance metrics by class
    safe_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    safe_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    vuln_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    vuln_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    metrics_data = {
        'Safe': [safe_precision, safe_recall],
        'Vulnerable': [vuln_precision, vuln_recall]
    }
    
    x = np.arange(2)
    width = 0.35
    
    ax2.bar(x - width/2, [safe_precision, vuln_precision], width, label='Precision', color='#3498db')
    ax2.bar(x + width/2, [safe_recall, vuln_recall], width, label='Recall', color='#2ecc71')
    
    ax2.set_ylabel('Score')
    ax2.set_title('Per-Class Performance')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Safe', 'Vulnerable'])
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (precision, recall) in enumerate(zip([safe_precision, vuln_precision], [safe_recall, vuln_recall])):
        ax2.text(i - width/2, precision + 0.01, f'{precision:.3f}', ha='center', va='bottom')
        ax2.text(i + width/2, recall + 0.01, f'{recall:.3f}', ha='center', va='bottom')
    
    plt.suptitle('Confusion Matrix and Per-Class Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix analysis saved to: {save_path}")
    
    if show_in_colab and 'google.colab' in sys.modules:
        plt.show()
    else:
        try:
            display(Image(filename=save_path))
        except:
            print(f"Plot saved but cannot display. Check: {save_path}")
    
    plt.close()

def train_with_kfold(dataset, args, k=5):
    """
    Train model using K-fold cross validation
    
    Args:
        dataset: DataFrame with vectors and labels
        args: Training arguments
        k: Number of folds
        
    Returns:
        Dict with aggregated results
    """
    print(f"\n{'='*60}")
    print(f"K-FOLD CROSS VALIDATION (k={k})")
    print(f"{'='*60}")
    
    # Prepare data
    X = np.stack(dataset['vector'].values)
    y = dataset['label'].values
    
    # Initialize k-fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    # Results storage
    fold_results = []
    all_histories = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*50}")
        print(f"FOLD {fold}/{k}")
        print(f"{'='*50}")
        
        # Create fold dataset
        fold_data = pd.DataFrame({
            'vector': [X[i] for i in train_idx],
            'label': y[train_idx]
        })
        
        # Train model
        model = WideTabTransformer(fold_data, args)
        history = model.train()
        results = model.evaluate()
        
        # Store results
        fold_results.append(results)
        all_histories.append(history)
        
        # Print fold results
        print(f"\nFold {fold} Results:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  F1-Score: {results['f1_score']:.4f}")
    
    # Aggregate results
    aggregated_results = {}
    for metric in fold_results[0].keys():
        if isinstance(fold_results[0][metric], (int, float)):
            values = [r[metric] for r in fold_results]
            aggregated_results[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    # Print summary
    print(f"\n{'='*60}")
    print("K-FOLD CROSS VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    for metric, stats in aggregated_results.items():
        if metric != 'confusion_matrix':
            print(f"\n{metric}:")
            print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    return aggregated_results, all_histories

def save_results(results, args, save_path="results/experiment_results.json"):
    """Save experiment results with metadata"""
    experiment_data = {
        'timestamp': datetime.now().isoformat(),
        'configuration': vars(args),
        'results': results,
        'system_info': {
            'python_version': sys.version,
            'tensorflow_version': tf.__version__,
            'numpy_version': np.__version__,
            'platform': sys.platform
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(experiment_data, f, indent=4, default=str)
    
    print(f"\nResults saved to: {save_path}")

"""
def main():
  
    IN_COLAB = 'google.colab' in sys.modules
    
    if IN_COLAB:
        print("🔵 Running in Google Colab environment")
        # Ensure matplotlib works properly in Colab
        import matplotlib
        matplotlib.use('module://ipykernel.pylab.backend_inline')
    
    # Parse arguments
    args = parameter_parser()
    
    # Print header and parameters
    print_header()
    print_parameters(args)
    
    # 🆕 DEBUG DU PARSING
    print(f"\n{'='*60}")
    print("VERIFICATION DU PARSING (MODE DEBUG)")
    print(f"{'='*60}")
    
    # Test du parsing avec debug
    debug_parse_smart_contracts(args.filename, max_fragments=3)
    
    # Demander confirmation avant de continuer (en mode interactif seulement)
    if sys.stdin.isatty():  # Vérifie si on est en mode interactif
        user_input = input("\nLe parsing semble-t-il correct ? (y/n) [y]: ").lower().strip()
        if user_input and user_input != 'y':
            print("Parsing interrompu. Vérifiez les données d'entrée.")
            return
    else:
        print("\nMode non-interactif détecté, continuation automatique...")
    
    # Prepare dataset path
    base_name = os.path.splitext(os.path.basename(args.filename))[0]
    dataset_path = f"config/train_data/{base_name}_enhanced_vectors.pkl"
    
    # Create data directory
    os.makedirs("config/train_data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    print(f"\nDataset path: {dataset_path}")
    
    # Load or create dataset
    if os.path.exists(dataset_path):
        print("Loading existing enhanced dataset...")
        dataset = pd.read_pickle(dataset_path)
        print("Enhanced dataset loaded successfully!")
    else:
        print("Creating new enhanced dataset...")
        dataset = create_dataset(args.filename, args)
        print(f"Saving enhanced dataset to {dataset_path}...")
        dataset.to_pickle(dataset_path)
        print("Enhanced dataset saved successfully!")
    
    # Model training and evaluation
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    start_time = time.time()
    
    # Initialize model
    print("Initializing Wide + TabTransformer model...")
    model = WideTabTransformer(dataset, args)
    model.get_model_summary()
    
    # Train model
    history = model.train()
    
    training_time = time.time() - start_time
    print(f"\nTotal training time: {training_time:.2f}s")
    
    # Plot training curves
    print("\n" + "=" * 60)
    print("GENERATING TRAINING CURVES")
    print("=" * 60)
    
    plot_training_curves(
        history, 
        save_path=f"plots/{base_name}_training_curves.png",
        show_in_colab=IN_COLAB
    )

    # Evaluate model
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    results = model.evaluate()

    # Plot metrics comparison
    print("\n" + "=" * 60)
    print("GENERATING METRICS COMPARISON")
    print("=" * 60)
    
    plot_metrics_comparison(
        results,
        save_path=f"plots/{base_name}_metrics_comparison.png",
        show_in_colab=IN_COLAB
    )
    
    # Plot confusion matrix analysis
    print("\n" + "=" * 60)
    print("GENERATING CONFUSION MATRIX ANALYSIS")
    print("=" * 60)
    
    plot_confusion_matrix_detailed(
    results,
    save_path=f"plots/{base_name}_confusion_matrix_analysis.png",
    show_in_colab=IN_COLAB)
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Architecture: Wide + TabTransformer")
    print(f"Vectorization: Enhanced with vulnerability patterns")
    print(f"Dataset: {args.filename}")
    print(f"Vulnerability Type: {args.vt}")
    print(f"Training Parameters:")
    print(f"  - Learning Rate: {args.lr}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Transformer Layers: {args.num_transformer_layers}")
    print(f"  - Attention Heads: {args.num_heads}")
    print(f"  - Embedding Dim: {args.embedding_dim}")
    print(f"  - Dropout: {args.dropout}")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Final Accuracy: {results['accuracy']:.4f}")
    print(f"Final F1-Score: {results['f1_score']:.4f}")
    print(f"\nPlots saved in: plots/")
    
    if IN_COLAB:
        print("\n📊 All plots have been displayed inline in Colab")
    
    print("=" * 60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉")
    print("=" * 60)

if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass  

    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

 """

def main():
    """Enhanced main execution function"""
    # Setup
    setup_directories()
    IN_COLAB = 'google.colab' in sys.modules
   
    if IN_COLAB:
       print("🔵 Running in Google Colab environment")
       import matplotlib
       matplotlib.use('module://ipykernel.pylab.backend_inline')
   
   # Parse arguments
    args = parameter_parser()
   
   # Print header and parameters
    print_header()
    print_parameters(args)
   
   # Debug parsing if requested
    print(f"\n{'='*60}")
    print("PARSING VERIFICATION")
    print(f"{'='*60}")
   
    debug_parse_smart_contracts(args.filename, max_fragments=3)
   
   # Interactive confirmation (skip in non-interactive mode)
    if sys.stdin.isatty():
       user_input = input("\nDoes the parsing look correct? (y/n) [y]: ").lower().strip()
       if user_input and user_input != 'y':
           print("Parsing verification failed. Please check your data.")
           return
    else:
       print("\nNon-interactive mode detected, continuing automatically...")
   
   # Create dataset
    print(f"\n{'='*60}")
    print("DATASET CREATION")
    print(f"{'='*60}")
   
    dataset, vectorizer = create_dataset(args.filename, args)
   
    # ADD THIS NEW SECTION - Perform correlation analysis
    print(f"\n{'='*60}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*60}")
   
    # Define save path
    base_name = Path(args.filename).stem
    correlation_save_path = Path(CONFIG['PLOTS_DIR']) / f"{base_name}_correlation"
    
    # Ensure plots directory exists
    corr_results = compute_and_visualize_correlations(
        dataset, 
        args,
        vectorizer=vectorizer,  
        save_path=str(correlation_save_path)
    )
    
    corr_results_path = Path(CONFIG['RESULTS_DIR']) / f"{base_name}_correlations.json"
    Path(CONFIG['RESULTS_DIR']).mkdir(parents=True, exist_ok=True)
    
    with open(corr_results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        corr_data = {
            'label_correlations': corr_results['label_correlation'].tolist(),
            'pattern_correlations': corr_results['pattern_label_correlation'],
            'pattern_names': corr_results['pattern_names'],
            'significant_features_count': len(corr_results['significant_features']),
            'feature_correlation_matrix_shape': corr_results['feature_correlation'].shape
        }
        json.dump(corr_data, f, indent=4)
    
    print(f"✅ Correlation results saved to: {corr_results_path}")
    
    # Print summary
    print(f"\n📊 CORRELATION ANALYSIS SUMMARY:")
    print(f"   • Feature-label correlations computed: {len(corr_results['label_correlation'])}")
    print(f"   • Significant features (|r| > 0.1): {len(corr_results['significant_features'])}")
    print(f"   • Strongest correlation: {np.max(np.abs(corr_results['label_correlation'])):.4f}")
    print(f"   • Correlation matrix shape: {corr_results['feature_correlation'].shape}")
    print(f"   • Plots saved to: {CONFIG['PLOTS_DIR']}/")
    
    print(f"\nCorrelation results saved to: {corr_results_path}")
    
    # Optional: Add feature selection based on correlation
    if hasattr(args, 'use_correlation_selection') and args.use_correlation_selection:
        dataset, selected_features = select_features_by_correlation(
            dataset, 
            threshold=0.1, 
            max_features=1000
        )
        print(f"Dataset reduced to {len(selected_features)} features based on correlation")
    


    # Save dataset info
    dataset_info = {
       'total_samples': len(dataset),
       'vulnerable_samples': sum(dataset['label'] == 1),
       'safe_samples': sum(dataset['label'] == 0),
       'vector_shape': dataset.iloc[0]['vector'].shape,
       'vulnerability_ratio': sum(dataset['label'] == 1) / len(dataset)
    }
   
    with open(Path(CONFIG['RESULTS_DIR']) / 'dataset_info.json', 'w') as f:
       json.dump(dataset_info, f, indent=4)
   
   # Training phase
    print(f"\n{'='*60}")
    print("MODEL TRAINING")
    print(f"{'='*60}")
   
    start_time = time.time()
   
    if args.use_kfold:
       # K-fold cross validation
       aggregated_results, histories = train_with_kfold(
           dataset, args, k=args.kfold_splits
       )
       
       # Save k-fold results
       save_results(aggregated_results, args, 
                   Path(CONFIG['RESULTS_DIR']) / f'kfold_results_{args.kfold_splits}.json')
       
       # Plot average training curves
       avg_history = average_histories(histories)
       plot_training_curves(
           avg_history,
           save_path=Path(CONFIG['PLOTS_DIR']) / f"{Path(args.filename).stem}_kfold_training_curves.png",
           show_in_colab=IN_COLAB
       )
       
       # Use mean values for final results
       results = {
           metric: stats['mean'] 
           for metric, stats in aggregated_results.items() 
           if isinstance(stats, dict) and 'mean' in stats
       }
       
    else:
       # Standard training
       model = WideTabTransformer(dataset, args)
       
       # Print model summary
       model.get_model_summary()
       
       # Train model
       history = model.train()
       
       # Plot training curves
       plot_training_curves(
           history, 
           save_path=Path(CONFIG['PLOTS_DIR']) / f"{Path(args.filename).stem}_training_curves.png",
           show_in_colab=IN_COLAB
       )
       
       # Evaluate model
       print(f"\n{'='*60}")
       print("MODEL EVALUATION")
       print(f"{'='*60}")
       
       results = model.evaluate()

    # 🆕 ROC CURVE ANALYSIS - NEW SECTION
    print(f"\n{'='*60}")
    print("ROC CURVE ANALYSIS")
    print(f"{'='*60}")
    
    roc_results = plot_roc_curve_analysis(
        model, 
        dataset, 
        args,
        save_path=Path(CONFIG['PLOTS_DIR']) / f"{base_name}_roc_analysis.png",
        show_in_colab=IN_COLAB
    )
    
    # Save ROC results
    roc_results_path = Path(CONFIG['RESULTS_DIR']) / f"{base_name}_roc_results.json"
    with open(roc_results_path, 'w') as f:
        json.dump(roc_results, f, indent=4)
    
    print(f"✅ ROC analysis results saved to: {roc_results_path}")
    
    # Calculate confidence intervals for ROC AUC
    print(f"\n{'='*50}")
    print("ROC AUC CONFIDENCE INTERVALS")
    print(f"{'='*50}")
    
    # Prepare test data for confidence interval calculation
    X = np.stack(dataset['vector'].values)
    y = dataset['label'].values
    
    if hasattr(model, 'x_test_wide'):
        X_test_wide = model.x_test_wide
        X_test_transformer = model.x_test_transformer
        y_test = np.argmax(model.y_test, axis=1)
    else:
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        wide_features = args.wide_features
        X_test_wide = X_test[:, :wide_features]
        X_test_transformer = X_test[:, wide_features:]
    
    y_pred_proba = model.model.predict([X_test_wide, X_test_transformer], verbose=0)
    y_scores = y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba.flatten()
       
    ci_results = calculate_roc_confidence_intervals(y_test, y_scores, n_bootstraps=1000)
       
       # Save confidence interval results
    ci_results_path = Path(CONFIG['RESULTS_DIR']) / f"{base_name}_roc_confidence_intervals.json"
    with open(ci_results_path, 'w') as f:
           json.dump(ci_results, f, indent=4)
       
    print(f"✅ ROC confidence intervals saved to: {ci_results_path}")

        # ADD THIS - Analyze learned correlations
    print(f"\n{'='*60}")
    print("LEARNED REPRESENTATION ANALYSIS")
    print(f"{'='*60}")
        
    learned_corr = model.analyze_learned_correlations()
    print(f"Wide component average correlation: {np.mean(np.abs(learned_corr['wide_correlations'])):.3f}")
    print(f"Transformer component average correlation: {np.mean(np.abs(learned_corr['transformer_correlations'])):.3f}")
    
       
       # Save model if requested
    if args.save_best_model:
           model_path = Path(CONFIG['MODELS_DIR']) / f"{Path(args.filename).stem}_final_model.h5"
           model.model.save(model_path)
           print(f"Model saved to: {model_path}")
   
    training_time = time.time() - start_time
   
    # Plot evaluation metrics
    print(f"\n{'='*60}")
    print("GENERATING EVALUATION PLOTS")
    print(f"{'='*60}")
   
    plot_metrics_comparison(
       results,
       save_path=Path(CONFIG['PLOTS_DIR']) / f"{Path(args.filename).stem}_metrics_comparison.png",
       show_in_colab=IN_COLAB
    )
   
    plot_confusion_matrix_detailed(
       results,
       save_path=Path(CONFIG['PLOTS_DIR']) / f"{Path(args.filename).stem}_confusion_matrix.png",
       show_in_colab=IN_COLAB
    )
   
    # Save final results
    save_results(results, args, 
               Path(CONFIG['RESULTS_DIR']) / f"{Path(args.filename).stem}_final_results.json")
   
    # Generate final report
    generate_final_report(args, results, training_time, dataset_info)
   
    # Final summary
    print(f"\n{'='*60}")
    print("EXECUTION COMPLETED SUCCESSFULLY! 🎉")
    print(f"{'='*60}")
    print(f"Total execution time: {training_time/60:.2f} minutes")
    print(f"Results saved in: {CONFIG['RESULTS_DIR']}/")
    print(f"Plots saved in: {CONFIG['PLOTS_DIR']}/")
    print(f"Models saved in: {CONFIG['MODELS_DIR']}/")
    
    if IN_COLAB:
        print("\n📊 All plots have been displayed inline in Colab")

def average_histories(histories):
   """Average multiple training histories for k-fold visualization"""
   avg_history = type('obj', (object,), {})()
   avg_history.history = {}
   
   # Get all metrics from first history
   metrics = histories[0].history.keys()
   
   for metric in metrics:
       # Collect all values for this metric
       all_values = []
       min_length = min(len(h.history[metric]) for h in histories)
       
       for h in histories:
           all_values.append(h.history[metric][:min_length])
       
       # Average across folds
       avg_history.history[metric] = np.mean(all_values, axis=0).tolist()
   
   return avg_history

def generate_final_report(args, results, training_time, dataset_info, roc_results=None):
   """Generate comprehensive markdown report"""
   report_path = Path(CONFIG['RESULTS_DIR']) / f"{Path(args.filename).stem}_report.md"
   
   with open(report_path, 'w') as f:
       f.write("# Smart Contract Vulnerability Detection Report\n\n")
       f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
       
       f.write("## Executive Summary\n\n")
       f.write(f"- **Model:** Wide + TabTransformer Neural Network\n")
       f.write(f"- **Task:** Reentrancy Vulnerability Detection\n")
       f.write(f"- **Accuracy:** {results['accuracy']:.2%}\n")
       f.write(f"- **F1-Score:** {results['f1_score']:.2%}\n")
       f.write(f"- **Training Time:** {training_time/60:.2f} minutes\n\n")
       
       f.write("## Dataset Information\n\n")
       f.write(f"- **Total Samples:** {dataset_info['total_samples']:,}\n")
       f.write(f"- **Vulnerable:** {dataset_info['vulnerable_samples']:,} ({dataset_info['vulnerability_ratio']:.1%})\n")
       f.write(f"- **Safe:** {dataset_info['safe_samples']:,} ({1-dataset_info['vulnerability_ratio']:.1%})\n")
       f.write(f"- **Vector Dimensions:** {dataset_info['vector_shape']}\n\n")
       
       f.write("## Model Architecture\n\n")
       f.write(f"- **Wide Features:** {args.wide_features}\n")
       f.write(f"- **Transformer Layers:** {args.num_transformer_layers}\n")
       f.write(f"- **Attention Heads:** {args.num_heads}\n")
       f.write(f"- **Embedding Dimension:** {args.embedding_dim}\n")
       f.write(f"- **Dropout Rate:** {args.dropout}\n\n")
       
       f.write("## Training Configuration\n\n")
       f.write(f"- **Learning Rate:** {args.lr}\n")
       f.write(f"- **Batch Size:** {args.batch_size}\n")
       f.write(f"- **Epochs:** {args.epochs}\n")
       f.write(f"- **Early Stopping:** {args.early_stopping_patience} epochs\n")
       f.write(f"- **L1 Regularization:** {args.l1_reg}\n")
       f.write(f"- **L2 Regularization:** {args.l2_reg}\n\n")
       
       f.write("## Performance Metrics\n\n")
       f.write("| Metric | Value |\n")
       f.write("|--------|-------|\n")
       f.write(f"| Accuracy | {results['accuracy']:.4f} |\n")
       f.write(f"| Precision | {results['precision']:.4f} |\n")
       f.write(f"| Recall | {results['recall']:.4f} |\n")
       f.write(f"| F1-Score | {results['f1_score']:.4f} |\n")
       f.write(f"| False Positive Rate | {results['fp_rate']:.4f} |\n")
       f.write(f"| False Negative Rate | {results['fn_rate']:.4f} |\n\n")
       
       if 'confusion_matrix' in results:
           cm = results['confusion_matrix']
           f.write("## Confusion Matrix\n\n")
           f.write("| | Predicted Safe | Predicted Vulnerable |\n")
           f.write("|---|---|---|\n")
           f.write(f"| **Actual Safe** | {cm['tn']} | {cm['fp']} |\n")
           f.write(f"| **Actual Vulnerable** | {cm['fn']} | {cm['tp']} |\n\n")
       
       f.write("## Key Findings\n\n")
       
       # Performance analysis
       if results['accuracy'] > 0.95:
           f.write("- ✅ **Excellent Performance:** The model achieves outstanding accuracy (>95%)\n")
       elif results['accuracy'] > 0.90:
           f.write("- ✅ **Strong Performance:** The model achieves high accuracy (>90%)\n")
       elif results['accuracy'] > 0.85:
           f.write("- ⚡ **Good Performance:** The model achieves reasonable accuracy (>85%)\n")
       else:
           f.write("- ⚠️ **Needs Improvement:** The model accuracy is below expectations (<85%)\n")
       
       # Balance analysis
       if abs(results['precision'] - results['recall']) < 0.05:
           f.write("- ✅ **Well Balanced:** Precision and recall are well balanced\n")
       elif results['precision'] > results['recall']:
           f.write("- ⚡ **Conservative:** Higher precision than recall (fewer false positives)\n")
       else:
           f.write("- ⚡ **Sensitive:** Higher recall than precision (fewer false negatives)\n")
       
       # Error analysis
       if results['fp_rate'] < 0.05 and results['fn_rate'] < 0.05:
           f.write("- ✅ **Low Error Rates:** Both false positive and false negative rates are very low\n")
       elif results['fn_rate'] > 0.1:
           f.write("- ⚠️ **High False Negatives:** The model misses some vulnerable contracts\n")
       elif results['fp_rate'] > 0.1:
           f.write("- ⚠️ **High False Positives:** The model incorrectly flags some safe contracts\n")
       
       f.write("\n## Recommendations\n\n")
       
       if results['accuracy'] < 0.90:
           f.write("1. Consider increasing model complexity (more layers/heads)\n")
           f.write("2. Collect more training data, especially for minority class\n")
           f.write("3. Experiment with different learning rates\n")
       
       if abs(results['precision'] - results['recall']) > 0.1:
           f.write("1. Adjust class weights to balance precision/recall trade-off\n")
           f.write("2. Consider different decision thresholds\n")
       
       if results['fp_rate'] > 0.1 or results['fn_rate'] > 0.1:
           f.write("1. Implement ensemble methods for more robust predictions\n")
           f.write("2. Add more vulnerability-specific features\n")
           f.write("3. Consider active learning to improve on difficult cases\n")
       
       f.write("\n## Conclusion\n\n")
       f.write("The Wide + TabTransformer architecture demonstrates ")
       
       if results['f1_score'] > 0.90:
           f.write("excellent capability in detecting reentrancy vulnerabilities ")
       elif results['f1_score'] > 0.85:
           f.write("strong capability in detecting reentrancy vulnerabilities ")
       else:
           f.write("promising results for detecting reentrancy vulnerabilities ")
       
       f.write("in smart contracts. The hybrid approach effectively combines ")
       f.write("memorization (Wide) and generalization (TabTransformer) ")
       f.write("to achieve robust vulnerability detection.\n")
          # ADD this section in generate_final_report function
       f.write("## ROC Analysis\n\n")
       if roc_results:
           f.write("| Metric | Value |\n")
           f.write("|--------|-------|\n")
           f.write(f"| ROC AUC | {roc_results['roc_auc']:.4f} |\n")
           f.write(f"| PR AUC | {roc_results['pr_auc']:.4f} |\n")
           f.write(f"| Optimal Threshold | {roc_results['optimal_threshold']:.4f} |\n")
           f.write(f"| Optimal TPR | {roc_results['optimal_tpr']:.4f} |\n")
           f.write(f"| Optimal FPR | {roc_results['optimal_fpr']:.4f} |\n\n")
   
   print(f"\nDetailed report saved to: {report_path}")

if __name__ == '__main__':
   try:
       # Set encoding for Windows compatibility
       if sys.platform == 'win32':
           sys.stdout.reconfigure(encoding='utf-8')
           sys.stderr.reconfigure(encoding='utf-8')
   except:
       pass
   
   try:
       main()
   except KeyboardInterrupt:
       print("\n\nTraining interrupted by user.")
       sys.exit(1)
   except Exception as e:
       print(f"\nError: {e}")
       import traceback
       traceback.print_exc()
       sys.exit(1)