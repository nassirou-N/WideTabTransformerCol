import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Dense, Dropout, LayerNormalization, MultiHeadAttention,
    Add, Activation
)
from tensorflow.keras.regularizers import l1_l2
import numpy as np

class TabTransformer(Layer):
    """
    Enhanced TabTransformer layer for processing tabular data with transformer architecture.
    Designed specifically for smart contract vulnerability detection with improved stability.
    """
    
    def __init__(self, 
                 num_transformer_layers=2,
                 num_heads=4,
                 embedding_dim=32,
                 mlp_hidden_dim=64,
                 dropout_rate=0.35,
                 use_layer_norm=True,
                 l1_reg=0.01,
                 l2_reg=0.01,
                 attention_dropout=0.05,
                 use_residual=True,
                 use_positional_encoding=True,
                 **kwargs):
        """
        Initialize TabTransformer with enhanced parameters
        
        Args:
            num_transformer_layers: Number of transformer blocks
            num_heads: Number of attention heads
            embedding_dim: Dimension of embeddings
            mlp_hidden_dim: Hidden dimension of feed-forward network
            dropout_rate: Dropout rate for regularization
            use_layer_norm: Whether to use layer normalization
            l1_reg: L1 regularization strength
            l2_reg: L2 regularization strength
            attention_dropout: Dropout rate specifically for attention
            use_residual: Whether to use residual connections
            use_positional_encoding: Whether to add positional encodings
        """
        super(TabTransformer, self).__init__(**kwargs)
        
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.attention_dropout = attention_dropout
        self.use_residual = use_residual
        self.use_positional_encoding = use_positional_encoding
        
        # Regularizer
        self.regularizer = l1_l2(l1=l1_reg, l2=l2_reg)
        
        # Input feature embedding with regularization
        self.feature_embedding = Dense(
            embedding_dim, 
            activation='relu',
            kernel_regularizer=self.regularizer,
            name='feature_embedding'
        )
        
        # Initialize transformer components
        self._build_transformer_layers()
        
        # Output projection
        self.output_projection = Dense(
            embedding_dim,
            kernel_regularizer=self.regularizer,
            name='output_projection'
        )
    
    def _build_transformer_layers(self):
        """Initialize transformer layers and components with enhanced stability"""
        self.attention_layers = []
        self.layer_norms_1 = []
        self.layer_norms_2 = []
        self.mlp_layers = []
        self.dropout_layers = []
        self.attention_dropout_layers = []
        
        for i in range(self.num_transformer_layers):
            # Multi-head attention with scaled initialization
            attention = MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.embedding_dim // self.num_heads,
                dropout=self.attention_dropout,
                kernel_regularizer=self.regularizer,
                name=f'multi_head_attention_{i}'
            )
            self.attention_layers.append(attention)
            
            # Layer normalizations
            if self.use_layer_norm:
                self.layer_norms_1.append(LayerNormalization(
                    epsilon=1e-6, 
                    name=f'layer_norm_1_{i}'
                ))
                self.layer_norms_2.append(LayerNormalization(
                    epsilon=1e-6, 
                    name=f'layer_norm_2_{i}'
                ))
            
            # MLP (Feed Forward Network) with careful initialization
            mlp = tf.keras.Sequential([
                Dense(
                    self.mlp_hidden_dim, 
                    activation='gelu',  # GELU activation for smoother gradients
                    kernel_regularizer=self.regularizer,
                    kernel_initializer='glorot_uniform',
                    name=f'mlp_dense_1_{i}'
                ),
                Dropout(self.dropout_rate),
                Dense(
                    self.embedding_dim,
                    kernel_regularizer=self.regularizer,
                    kernel_initializer='glorot_uniform',
                    name=f'mlp_dense_2_{i}'
                )
            ], name=f'mlp_{i}')
            self.mlp_layers.append(mlp)
            
            # Dropout layers
            self.dropout_layers.append(Dropout(
                self.dropout_rate, 
                name=f'dropout_{i}'
            ))
            self.attention_dropout_layers.append(Dropout(
                self.attention_dropout, 
                name=f'attention_dropout_{i}'
            ))
    
    def _get_positional_encoding(self, seq_len, d_model):
        """Generate positional encoding for sequence"""
        position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * 
                          -(tf.math.log(10000.0) / d_model))
        
        pos_encoding = tf.zeros((seq_len, d_model))
        indices = tf.stack([
            tf.repeat(tf.range(seq_len), d_model // 2),
            tf.tile(tf.range(0, d_model, 2), [seq_len])
        ], axis=1)
        
        values = tf.reshape(tf.sin(position * div_term), [-1])
        pos_encoding = tf.tensor_scatter_nd_update(pos_encoding, indices, values)
        
        indices = tf.stack([
            tf.repeat(tf.range(seq_len), d_model // 2),
            tf.tile(tf.range(1, d_model, 2), [seq_len])
        ], axis=1)
        
        values = tf.reshape(tf.cos(position * div_term), [-1])
        pos_encoding = tf.tensor_scatter_nd_update(pos_encoding, indices, values)
        
        return pos_encoding[tf.newaxis, ...]

    def call(self, inputs, training=None, mask=None, **kwargs):
        """
        Forward pass through TabTransformer with enhanced stability
        
        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, feature_dim)
            training: Boolean indicating training mode
            mask: Optional attention mask
            
        Returns:
            Transformed features of shape (batch_size, sequence_length, embedding_dim)
        """
        # Project input features to embedding dimension
        x = self.feature_embedding(inputs)
        
        # Add positional encoding if enabled
        if self.use_positional_encoding:
            seq_len = tf.shape(inputs)[1]
            pos_encoding = self._get_positional_encoding(seq_len, self.embedding_dim)
            x = x + pos_encoding
        
        # Apply transformer layers with gradient-friendly operations
        for i in range(self.num_transformer_layers):
            # Store input for residual connection
            residual = x
            
            # Multi-head attention block
            attention_output = self.attention_layers[i](
                x, x, 
                training=training,
                attention_mask=mask
            )
            attention_output = self.attention_dropout_layers[i](
                attention_output, 
                training=training
            )
            
            # First residual connection
            if self.use_residual:
                x = Add()([residual, attention_output])
            else:
                x = attention_output
            
            # First layer normalization
            if self.use_layer_norm:
                x = self.layer_norms_1[i](x)
            
            # Store for second residual
            residual = x
            
            # MLP block
            mlp_output = self.mlp_layers[i](x, training=training)
            mlp_output = self.dropout_layers[i](mlp_output, training=training)
            
            # Second residual connection
            if self.use_residual:
                x = Add()([residual, mlp_output])
            else:
                x = mlp_output
            
            # Second layer normalization
            if self.use_layer_norm:
                x = self.layer_norms_2[i](x)
            
            # Add slight noise for regularization during training
            if training:
                noise = tf.random.normal(
                    shape=tf.shape(x), 
                    mean=0.0, 
                    stddev=0.005
                )
                x = x + noise
        
        # Final projection
        x = self.output_projection(x)
        
        return x
    
    def compute_attention_scores(self, inputs):
        """
        Compute and return attention scores for interpretability
        
        Args:
            inputs: Input tensor
            
        Returns:
            List of attention score matrices for each layer
        """
        x = self.feature_embedding(inputs)
        attention_scores = []
        
        for i in range(self.num_transformer_layers):
            # Get attention scores from multi-head attention
            _, scores = self.attention_layers[i](
                x, x, 
                return_attention_scores=True,
                training=False
            )
            attention_scores.append(scores)
            
            # Continue forward pass
            attention_output = self.attention_layers[i](x, x, training=False)
            x = Add()([x, attention_output])
            if self.use_layer_norm:
                x = self.layer_norms_1[i](x)
            
            mlp_output = self.mlp_layers[i](x, training=False)
            x = Add()([x, mlp_output])
            if self.use_layer_norm:
                x = self.layer_norms_2[i](x)
        
        return attention_scores

    def get_config(self):
        """Get layer configuration for serialization"""
        config = super(TabTransformer, self).get_config()
        config.update({
            "num_transformer_layers": self.num_transformer_layers,
            "num_heads": self.num_heads,
            "embedding_dim": self.embedding_dim,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "dropout_rate": self.dropout_rate,
            "use_layer_norm": self.use_layer_norm,
            "l1_reg": self.l1_reg,
            "l2_reg": self.l2_reg,
            "attention_dropout": self.attention_dropout,
            "use_residual": self.use_residual,
            "use_positional_encoding": self.use_positional_encoding
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create layer from configuration"""
        return cls(**config)


class ImprovedTabTransformer(TabTransformer):
    """
    Extended TabTransformer with additional features for better performance
    """
    
    def __init__(self, 
                 use_gating=True,
                 use_feature_interaction=True,
                 interaction_dropout=0.2,
                 **kwargs):
        """
        Initialize improved TabTransformer
        
        Args:
            use_gating: Whether to use gating mechanism
            use_feature_interaction: Whether to use explicit feature interactions
            interaction_dropout: Dropout rate for feature interactions
        """
        self.use_gating = use_gating
        self.use_feature_interaction = use_feature_interaction
        self.interaction_dropout = interaction_dropout
        
        super().__init__(**kwargs)
        
        # Additional components
        if self.use_gating:
            self.gating_layers = []
            for i in range(self.num_transformer_layers):
                gate = Dense(
                    self.embedding_dim,
                    activation='sigmoid',
                    kernel_regularizer=self.regularizer,
                    name=f'gating_{i}'
                )
                self.gating_layers.append(gate)
        
        if self.use_feature_interaction:
            self.interaction_layer = Dense(
                self.embedding_dim,
                activation='relu',
                kernel_regularizer=self.regularizer,
                name='feature_interaction'
            )
            self.interaction_dropout = Dropout(interaction_dropout)
    
    def call(self, inputs, training=None, mask=None, **kwargs):
        """Enhanced forward pass with gating and feature interactions"""
        # Project input features
        x = self.feature_embedding(inputs)
        
        # Add feature interactions if enabled
        if self.use_feature_interaction:
            # Compute pairwise interactions
            batch_size = tf.shape(inputs)[0]
            seq_len = tf.shape(inputs)[1]
            
            # Reshape for interaction computation
            x_expanded = tf.expand_dims(x, 2)  # (B, L, 1, D)
            x_tiled = tf.tile(x_expanded, [1, 1, seq_len, 1])  # (B, L, L, D)
            x_transpose = tf.transpose(x_tiled, [0, 2, 1, 3])  # (B, L, L, D)
            
            # Compute interactions
            interactions = x_tiled * x_transpose  # (B, L, L, D)
            interactions = tf.reduce_mean(interactions, axis=2)  # (B, L, D)
            
            # Add interaction features
            interaction_features = self.interaction_layer(interactions)
            interaction_features = self.interaction_dropout(
                interaction_features, 
                training=training
            )
            x = x + interaction_features
        
        # Add positional encoding if enabled
        if self.use_positional_encoding:
            seq_len = tf.shape(inputs)[1]
            pos_encoding = self._get_positional_encoding(seq_len, self.embedding_dim)
            x = x + pos_encoding
        
        # Apply transformer layers with gating
        for i in range(self.num_transformer_layers):
            residual = x
            
            # Multi-head attention
            attention_output = self.attention_layers[i](
                x, x, 
                training=training,
                attention_mask=mask
            )
            attention_output = self.attention_dropout_layers[i](
                attention_output, 
                training=training
            )
            
            # Apply gating if enabled
            if self.use_gating:
                gate = self.gating_layers[i](x)
                attention_output = gate * attention_output
            
            # Residual connection
            if self.use_residual:
                x = Add()([residual, attention_output])
            else:
                x = attention_output
            
            if self.use_layer_norm:
                x = self.layer_norms_1[i](x)
            
            # MLP block
            residual = x
            mlp_output = self.mlp_layers[i](x, training=training)
            mlp_output = self.dropout_layers[i](mlp_output, training=training)
            
            if self.use_residual:
                x = Add()([residual, mlp_output])
            else:
                x = mlp_output
            
            if self.use_layer_norm:
                x = self.layer_norms_2[i](x)
        
        # Final projection
        x = self.output_projection(x)
        
        return x
    
    def get_config(self):
        """Get configuration including improved features"""
        config = super().get_config()
        config.update({
            "use_gating": self.use_gating,
            "use_feature_interaction": self.use_feature_interaction,
            "interaction_dropout": self.interaction_dropout
        })
        return config