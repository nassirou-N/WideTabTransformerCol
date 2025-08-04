import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import warnings
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Normalization, Concatenate, Flatten, Dropout, Dense, Input, 
    GlobalAveragePooling1D, BatchNormalization, GaussianNoise
)
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, 
    TensorBoard, LearningRateScheduler
)
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import datetime

from config.tabtransformer.TabTransformer import TabTransformer

# Suppress warnings and set seeds
warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

class WideTabTransformer:
    """
    Wide + TabTransformer model for smart contract vulnerability detection.
    
    Architecture:
    - Wide component: Shallow processing of first N features with regularization
    - TabTransformer component: Deep transformer processing of remaining features
    - Fusion: Concatenation and final classification with stability improvements
    """
    
    def __init__(self, data, args):
        self.args = args
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Prepare data
        self._prepare_data(data)
        
        # Build model
        self.model = self._build_model()
        
        # Initialize best metrics
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        
        print(f"Model initialized with {len(self.x_train_wide)} training samples")
        print(f"Wide features: {self.x_train_wide.shape[1:]}")
        print(f"TabTransformer features: {self.x_train_transformer.shape[1:]}")
    
    def _prepare_data(self, data):
        """Prepare and split data for training with enhanced preprocessing"""
        self.vectors = np.stack(data.iloc[:, 0].values)
        self.labels = data.iloc[:, 1].values
        
        # Get balanced indices
        positive_idxs = np.where(self.labels == 1)[0]
        negative_idxs = np.where(self.labels == 0)[0]
        
        print(f"Original distribution - Safe: {len(negative_idxs)}, Vulnerable: {len(positive_idxs)}")
        
        # Apply SMOTE if enabled and imbalanced
        if self.args.use_smote and len(positive_idxs) < len(negative_idxs) * 0.8:
            print("Applying SMOTE for class balancing...")
            vectors_flat = self.vectors.reshape(len(self.vectors), -1)
            smote = SMOTE(random_state=42, k_neighbors=min(5, len(positive_idxs)-1))
            vectors_resampled, labels_resampled = smote.fit_resample(vectors_flat, self.labels)
            self.vectors = vectors_resampled.reshape(-1, self.vectors.shape[1], self.vectors.shape[2])
            self.labels = labels_resampled
            print(f"After SMOTE - Total samples: {len(self.labels)}")
        
        # Create indices for splitting
        idxs = np.arange(len(self.labels))
        
        # Train/test split with stratification
        x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
            self.vectors, self.labels, idxs,
            test_size=0.2, stratify=self.labels, random_state=42
        )
        
        # Further split train into train/val
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train,
            test_size=0.2, stratify=y_train, random_state=42
        )
        
        # Split features for Wide and TabTransformer components
        wide_features = self.args.wide_features
        self.x_train_wide = x_train[:, :wide_features]
        self.x_train_transformer = x_train[:, wide_features:]
        self.x_val_wide = x_val[:, :wide_features]
        self.x_val_transformer = x_val[:, wide_features:]
        self.x_test_wide = x_test[:, :wide_features]
        self.x_test_transformer = x_test[:, wide_features:]
        
        # Convert labels to categorical
        self.y_train = to_categorical(y_train)
        self.y_val = to_categorical(y_val)
        self.y_test = to_categorical(y_test)
        
        # Calculate class weights for balanced training
        classes = np.array([0, 1])
        class_weights = compute_class_weight(
            class_weight='balanced', classes=classes, y=self.labels
        )
        self.class_weight = {index: weight for index, weight in enumerate(class_weights)}
        
        # Optimal weight adjustment
        if self.args.complexity_weighting:
            # Calculate optimal weight ratio
            pos_ratio = len(positive_idxs) / len(self.labels)
            weight_multiplier = min(2.0, 1.0 / pos_ratio) if pos_ratio < 0.5 else 1.0
            self.class_weight[1] *= weight_multiplier
            # Cap the maximum weight to prevent instability
            self.class_weight[1] = min(self.class_weight[1], 3.0)
        
        print(f"Class distribution - Train: Safe: {sum(y_train == 0)}, Vulnerable: {sum(y_train == 1)}")
        print(f"Class distribution - Val: Safe: {sum(y_val == 0)}, Vulnerable: {sum(y_val == 1)}")
        print(f"Class distribution - Test: Safe: {sum(y_test == 0)}, Vulnerable: {sum(y_test == 1)}")
        print(f"Class weights: {self.class_weight}")
    
    def _build_model(self):
        """Build Wide + TabTransformer model with enhanced regularization"""
        # Define regularizer
        regularizer = l1_l2(l1=self.args.l1_reg, l2=self.args.l2_reg)
        
        # Define inputs
        input_wide = Input(shape=self.x_train_wide.shape[1:], name='wide_input')
        input_transformer = Input(shape=self.x_train_transformer.shape[1:], name='transformer_input')
        
        # Wide component with enhanced regularization
        wide = Normalization(name='wide_normalization')(input_wide)
        
        # Add slight noise for regularization
        if self.args.use_data_augmentation:
            wide = GaussianNoise(0.01)(wide)
        
        wide_flattened = Flatten(name='wide_flatten')(wide)
        
        # Add batch normalization
        wide_flattened = BatchNormalization(name='wide_batch_norm')(wide_flattened)
        
        # Add a dense layer for wide component
        wide_processed = Dense(
            128, 
            activation='relu', 
            kernel_regularizer=regularizer,
            name='wide_dense'
        )(wide_flattened)
        wide_processed = Dropout(self.args.dropout)(wide_processed)
        
        # TabTransformer component
        transformer_input = Normalization(name='transformer_normalization')(input_transformer)
        
        # Add slight noise for regularization
        if self.args.use_data_augmentation:
            transformer_input = GaussianNoise(0.01)(transformer_input)
        
        # Apply TabTransformer
        tabtransformer = TabTransformer(
            num_transformer_layers=self.args.num_transformer_layers,
            num_heads=self.args.num_heads,
            embedding_dim=self.args.embedding_dim,
            mlp_hidden_dim=self.args.mlp_hidden_dim,
            dropout_rate=self.args.dropout,
            use_layer_norm=True
        )(transformer_input)
        
        # Global average pooling to aggregate sequence
        transformer_pooled = GlobalAveragePooling1D(name='transformer_pooling')(tabtransformer)
        
        # Additional processing for transformer output
        transformer_dense = Dense(
            128, 
            activation='relu', 
            kernel_regularizer=regularizer,
            name='transformer_dense'
        )(transformer_pooled)
        transformer_dense = BatchNormalization(name='transformer_batch_norm')(transformer_dense)
        transformer_dense = Dropout(self.args.dropout, name='transformer_dropout')(transformer_dense)
        
        # Fusion: Concatenate wide and transformer components
        merged = Concatenate(axis=-1, name='fusion')([wide_processed, transformer_dense])
        
        # Final classification layers with enhanced regularization
        dense_1 = Dense(
            256, 
            activation='relu', 
            kernel_regularizer=regularizer,
            name='dense_1'
        )(merged)
        dense_1 = BatchNormalization(name='batch_norm_1')(dense_1)
        dense_1 = Dropout(self.args.dropout, name='dropout_1')(dense_1)
        
        dense_2 = Dense(
            128, 
            activation='relu', 
            kernel_regularizer=regularizer,
            name='dense_2'
        )(dense_1)
        dense_2 = BatchNormalization(name='batch_norm_2')(dense_2)
        dense_2 = Dropout(self.args.dropout, name='dropout_2')(dense_2)
        
        # Output layer
        output = Dense(2, activation='softmax', name='output')(dense_2)
        
        # Create and compile model
        model = Model(inputs=[input_wide, input_transformer], outputs=output, name='WideTabTransformer')
        
        # Use gradient clipping for stability
        optimizer = Adam(
            learning_rate=self.lr,
            clipnorm=self.args.gradient_clip
        )
        
        # Compile with additional metrics
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        return model
    
    def _get_callbacks(self):
        """Get training callbacks for better model training"""
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.args.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.args.reduce_lr_patience,
            min_lr=self.args.min_lr,
            verbose=1,
            mode='min'
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint
        if self.args.save_best_model:
            checkpoint = ModelCheckpoint(
                filepath=self.args.model_checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                mode='max'
            )
            callbacks.append(checkpoint)
        
        # TensorBoard
        if self.args.tensorboard:
            log_dir = os.path.join('logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard = TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq='epoch'
            )
            callbacks.append(tensorboard)
        
        # Custom learning rate scheduler
        # Cosine annealing learning rate scheduler
        def cosine_annealing_schedule(epoch):
            """Cosine annealing learning rate schedule"""
            initial_lr = self.lr
            final_lr = self.args.min_lr
            return final_lr + (initial_lr - final_lr) * 0.5 * (1 + np.cos(np.pi * epoch / self.epochs))
        
        cosine_scheduler = LearningRateScheduler(cosine_annealing_schedule, verbose=1)
        callbacks.append(cosine_scheduler)

        def lr_schedule(epoch):
            """Learning rate schedule"""
            lr = self.lr
            if epoch > 100:
                lr *= 0.8
            elif epoch > 200:
                lr *= 0.6
            elif epoch > 350:
                lr *= 0.4
            return lr
        
        lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
        callbacks.append(lr_scheduler)
        
        return callbacks
    
    def train(self):
        """Train the model with enhanced training strategy"""
        print("\nStarting training...")
        print(f"Epochs: {self.epochs}, Batch size: {self.batch_size}, Learning rate: {self.lr}")
        
        if self.args.progressive_training:
            return self._progressive_train()
        
        # Get callbacks
        callbacks = self._get_callbacks()
        
        # Train model
        history = self.model.fit(
            [self.x_train_wide, self.x_train_transformer], 
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=self.class_weight,
            validation_data=([self.x_val_wide, self.x_val_transformer], self.y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        
        # Save training history
        self._save_training_history(history)
        
        return history
    
    def _progressive_train(self):
        """Progressive training strategy: train components separately then together"""
        print("\nUsing progressive training strategy...")
        
        # Phase 1: Train only Wide component
        print("\nPhase 1: Training Wide component...")
        for layer in self.model.layers:
            if 'transformer' in layer.name or 'tabtransformer' in layer.name.lower():
                layer.trainable = False
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        history_phase1 = self.model.fit(
            [self.x_train_wide, self.x_train_transformer], 
            self.y_train,
            epochs=10,
            batch_size=self.batch_size,
            class_weight=self.class_weight,
            validation_data=([self.x_val_wide, self.x_val_transformer], self.y_val),
            verbose=1
        )
        
        # Phase 2: Unfreeze and train everything
        print("\nPhase 2: Training full model...")
        for layer in self.model.layers:
            layer.trainable = True
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.lr * 0.1),  # Lower LR for fine-tuning
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        callbacks = self._get_callbacks()
        
        history_phase2 = self.model.fit(
            [self.x_train_wide, self.x_train_transformer], 
            self.y_train,
            epochs=self.epochs - 10,
            batch_size=self.batch_size,
            class_weight=self.class_weight,
            validation_data=([self.x_val_wide, self.x_val_transformer], self.y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        history = self._combine_histories(history_phase1, history_phase2)
        
        print("Progressive training completed!")
        return history
    
    def _combine_histories(self, hist1, hist2):
        """Combine two training histories"""
        combined = type('obj', (object,), {})()
        combined.history = {}
        
        for key in hist1.history.keys():
            combined.history[key] = hist1.history[key] + (hist2.history[key] if key in hist2.history else [])
        
        for key in hist2.history.keys():
            if key not in combined.history:
                combined.history[key] = [0] * len(hist1.history['loss']) + hist2.history[key]
        
        return combined
    
    def _save_training_history(self, history):
        """Save training history to file"""
        import json
        
        history_path = 'models/training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history.history, f, indent=4)
        print(f"Training history saved to {history_path}")
    
    def evaluate(self):
        """Evaluate model performance with detailed metrics"""
        print("\nEvaluating model...")
        
        # Get predictions
        predictions = self.model.predict(
            [self.x_test_wide, self.x_test_transformer], 
            batch_size=self.batch_size, 
            verbose=0
        )
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(self.y_test, axis=1)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(true_classes, predicted_classes).ravel()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Print results
        print(f"\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
        print(f"False Positive Rate: {fp_rate:.4f}")
        print(f"False Negative Rate: {fn_rate:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"TN: {tn}, FP: {fp}")
        print(f"FN: {fn}, TP: {tp}")
        
        # Detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(true_classes, predicted_classes, 
                                  target_names=['Safe', 'Vulnerable']))
        
        # Save evaluation results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'fp_rate': fp_rate,
            'fn_rate': fn_rate,
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp),
                'fn': int(fn), 'tp': int(tp)
            }
        }
        
        import json
        with open('models/evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        return results
    
    def predict_single(self, contract_vector):
        """Predict vulnerability for a single contract"""
        # Prepare input
        wide_input = contract_vector[:self.args.wide_features].reshape(1, -1)
        transformer_input = contract_vector[self.args.wide_features:].reshape(1, -1)
        
        # Get prediction
        prediction = self.model.predict([wide_input, transformer_input], verbose=0)
        
        # Get class and confidence
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        return {
            'class': 'Vulnerable' if predicted_class == 1 else 'Safe',
            'confidence': float(confidence),
            'probabilities': {
                'safe': float(prediction[0][0]),
                'vulnerable': float(prediction[0][1])
            }
        }
    
    def get_model_summary(self):
        """Print model architecture summary"""
        print("\nModel Architecture:")
        self.model.summary()
        
        print(f"\nTabTransformer Configuration:")
        print(f"- Transformer layers: {self.args.num_transformer_layers}")
        print(f"- Attention heads: {self.args.num_heads}")
        print(f"- Embedding dimension: {self.args.embedding_dim}")
        print(f"- MLP hidden dimension: {self.args.mlp_hidden_dim}")
        print(f"- Wide features: {self.args.wide_features}")
        print(f"- Dropout rate: {self.args.dropout}")
        print(f"- L1 regularization: {self.args.l1_reg}")
        print(f"- L2 regularization: {self.args.l2_reg}")
        
        # Count parameters
        trainable_params = sum([np.prod(v.shape) for v in self.model.trainable_weights])
        non_trainable_params = sum([np.prod(v.shape) for v in self.model.non_trainable_weights])
        
        print(f"\nTotal parameters: {trainable_params + non_trainable_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
    
    # In wide_tabtransformer.py - Add inside WideTabTransformer class

   def analyze_learned_correlations(self):
    """
    Analyze correlations in the learned representations
    """
    from tensorflow.keras import Model
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    
    # Get intermediate representations
    # Wide model only needs wide input
    wide_model = Model(inputs=self.model.input[0], 
                       outputs=self.model.get_layer('wide_dense').output)
    
    # Transformer model only needs transformer input
    transformer_model = Model(inputs=self.model.input[1], 
                             outputs=self.model.get_layer('transformer_dense').output)
    
    # Get representations for validation data
    # Pass only the wide input to wide model
    wide_repr = wide_model.predict(self.x_val_wide)
    
    # Pass only the transformer input to transformer model
    transformer_repr = transformer_model.predict(self.x_val_transformer)
    
    # Compute correlations between representations and labels
    y_val_binary = np.argmax(self.y_val, axis=1)
    
    wide_correlations = []
    transformer_correlations = []
    
    for i in range(wide_repr.shape[1]):
        corr, _ = pearsonr(wide_repr[:, i], y_val_binary)
        wide_correlations.append(corr)
    
    for i in range(transformer_repr.shape[1]):
        corr, _ = pearsonr(transformer_repr[:, i], y_val_binary)
        transformer_correlations.append(corr)
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(wide_correlations, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Correlation with Label')
    plt.ylabel('Frequency')
    plt.title('Wide Component - Learned Feature Correlations')
    
    plt.subplot(1, 2, 2)
    plt.hist(transformer_correlations, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Correlation with Label')
    plt.ylabel('Frequency')
    plt.title('Transformer Component - Learned Feature Correlations')
    
    plt.tight_layout()
    
    # Create models directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)
    
    plt.savefig('models/learned_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'wide_correlations': wide_correlations,
        'transformer_correlations': transformer_correlations
    }