import argparse

def parameter_parser():
    """
    Enhanced parameter parser for Smart Contract Vulnerability Detection 
    using Wide + TabTransformer with improved vectorization and training stability
    """
    parser = argparse.ArgumentParser(
        description='Smart Contract Vulnerability Detection Using Wide + TabTransformer Neural Network'
    )

    # Required arguments
    parser.add_argument('filename', type=str, 
                       help="Path to smart contract file to process")
    parser.add_argument('-vt', type=str, choices=['ts', 're', 'io'], 
                       help="Vulnerability type: ts(timestamp), re(reentrancy), io(integer overflow)")
    
    # Training hyperparameters - OPTIMIZED VALUES
    parser.add_argument('--lr', type=float, default=0.00008,  # Reduced from 0.001
                       help='Learning rate (default: 0.00008)')
    parser.add_argument('--dropout', type=float, default=0.25,  # Increased from 0.2
                       help='Dropout rate (default: 0.25)')
    parser.add_argument('--epochs', type=int, default=500,  # Reduced from 20
                       help='Number of training epochs (default: 500)')
    parser.add_argument('--batch_size', type=int, default=48,  # Increased from 4
                       help='Batch size (default: 48)')
    
    # Early stopping and learning rate scheduling
    parser.add_argument('--early_stopping_patience', type=int, default=25,
                       help='Early stopping patience (default: 25)')
    parser.add_argument('--reduce_lr_patience', type=int, default=10,
                       help='Reduce LR on plateau patience (default: 10)')
    parser.add_argument('--min_lr', type=float, default=1e-7,
                       help='Minimum learning rate (default: 1e-7)')
    
    # Enhanced vectorization parameters
    parser.add_argument('--vec_length', type=int, default=128,  # Reduced from 150
                       help='Word2Vec vector dimension (default: 128)')
    parser.add_argument('--w2v_window', type=int, default=10,  # Increased from 8
                       help='Word2Vec context window size (default: 10)')
    parser.add_argument('--w2v_min_count', type=int, default=3,  # Increased from 2
                       help='Word2Vec minimum token frequency (default: 3)')
    parser.add_argument('--w2v_epochs', type=int, default=30,  # Increased from 20
                       help='Word2Vec training epochs (default: 30)')
    parser.add_argument('--w2v_negative', type=int, default=15,  # Increased from 10
                       help='Word2Vec negative sampling (default: 15)')
    parser.add_argument('--w2v_sample', type=float, default=1e-5,
                       help='Word2Vec subsampling threshold (default: 1e-5)')
    
    # Model architecture parameters - OPTIMIZED
    parser.add_argument('--wide_features', type=int, default=30,  # Reduced from 50
                       help='Number of features for wide component (default: 30)')
    
    # TabTransformer architecture parameters - SIMPLIFIED
    parser.add_argument('--num_transformer_layers', type=int, default=3,  # Reduced from 3
                       help='Number of transformer layers (default: 3)')
    parser.add_argument('--num_heads', type=int, default=8,  # Reduced from 8
                       help='Number of attention heads (default: 8)')
    parser.add_argument('--embedding_dim', type=int, default=64,  # Reduced from 64
                       help='Transformer embedding dimension (default: 64)')
    parser.add_argument('--mlp_hidden_dim', type=int, default=128,  # Reduced from 128
                       help='MLP hidden dimension in transformer (default: 128)')
    
    # Regularization parameters
    parser.add_argument('--l1_reg', type=float, default=0.005,
                       help='L1 regularization strength (default: 0.005)')
    parser.add_argument('--l2_reg', type=float, default=0.005,
                       help='L2 regularization strength (default: 0.005)')
    parser.add_argument('--gradient_clip', type=float, default=0.5,
                       help='Gradient clipping value (default: 0.5)')
    
    # Advanced options
    parser.add_argument('--use_enhanced_vectorization', action='store_true', default=True,
                       help='Use enhanced vectorization (default: True)')
    parser.add_argument('--use_data_augmentation', action='store_true', default=True,
                       help='Use data augmentation for smart contracts (default: True)')
    parser.add_argument('--use_smote', action='store_true', default=False,
                       help='Use SMOTE for class balancing (default: False)')
    parser.add_argument('--use_kfold', action='store_true', default=False,
                       help='Use K-fold cross validation (default: False)')
    parser.add_argument('--kfold_splits', type=int, default=5,
                       help='Number of K-fold splits (default: 5)')
    parser.add_argument('--progressive_training', action='store_true', default=False,
                       help='Use progressive training strategy (default: False)')
    parser.add_argument('--complexity_weighting', action='store_true', default=True,
                       help='Use complexity-based sample weighting (default: True)')
    parser.add_argument('--vocab_stats', action='store_true', default=False,
                       help='Print detailed vocabulary statistics (default: False)')
    
    # Monitoring and debugging
    parser.add_argument('--tensorboard', action='store_true', default=False,
                       help='Use TensorBoard logging (default: False)')
    parser.add_argument('--save_best_model', action='store_true', default=True,
                       help='Save best model during training (default: True)')
    parser.add_argument('--model_checkpoint_path', type=str, default='models/best_model.h5',
                       help='Path to save best model (default: models/best_model.h5)')
    
    # In arg_parser.py - Add after the advanced options section
    
    # Correlation analysis options
    parser.add_argument('--use_correlation_selection', action='store_true', default=False,
                       help='Use correlation-based feature selection (default: False)')
    parser.add_argument('--correlation_threshold', type=float, default=0.1,
                       help='Minimum correlation threshold for feature selection (default: 0.1)')
    parser.add_argument('--max_correlation_features', type=int, default=1000,
                       help='Maximum features to select based on correlation (default: 1000)')
    parser.add_argument('--analyze_correlations', action='store_true', default=True,
                       help='Perform correlation analysis (default: True)')

    return parser.parse_args()