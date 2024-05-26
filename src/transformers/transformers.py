import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


class Patch(layers.Layer):
    def __init__(self, patch_size):
        super(Patch, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches



class PatchEmbedding(layers.Layer):
    def __init__(self, num_patches, embedding_dim):
        super(PatchEmbedding, self).__init__()
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim
        self.projection = layers.Dense(embedding_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=embedding_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        embedded_patches = self.projection(patches)
        embedded_patches += self.position_embedding(positions)
        return embedded_patches



class MLP(layers.Layer):
    def __init__(self, hidden_units, dropout_rate):
        super(MLP, self).__init__()
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.dense_layers = [layers.Dense(units, activation=tf.nn.gelu) for units in hidden_units]
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x):
        for dense in self.dense_layers:
            x = dense(x)
        return self.dropout(x)
    


class ClassToken(layers.Layer):
    def __init__(self, embedding_dim):
        super(ClassToken, self).__init__()
        self.class_token = tf.Variable(tf.zeros((1, 1, embedding_dim)), trainable=True)

    def call(self, embeddings):
        batch_size = tf.shape(embeddings)[0]
        class_tokens = tf.broadcast_to(self.class_token, [batch_size, 1, embeddings.shape[-1]])
        return tf.concat([class_tokens, embeddings], axis=1)
    



class TransformerEncoderLayer(layers.Layer):
    def __init__(self, embedding_dim, num_heads, mlp_dim, dropout_rate):
        super(TransformerEncoderLayer, self).__init__()
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.multi_head_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = MLP([mlp_dim, embedding_dim], dropout_rate)

    def call(self, x, training):
        # Multi-head Self Attention
        attn_output = self.multi_head_attention(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.layer_norm1(x + attn_output)
        
        # MLP
        mlp_output = self.mlp(x)
        x = self.layer_norm2(x + mlp_output)
        return x



class ViTModel(Model):
    def __init__(self, image_size, patch_size, num_layers, num_classes, d_model, num_heads, mlp_dim, dropout_rate):
        super(ViTModel, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.patch = Patch(patch_size)
        self.patch_embedding = PatchEmbedding(self.num_patches, d_model)
        self.class_token = ClassToken(d_model)
        self.encoder_layers = [TransformerEncoderLayer(d_model, num_heads, mlp_dim, dropout_rate) for _ in range(num_layers)]
        self.mlp_head = layers.Dense(num_classes)

    def call(self, images, training):
        # Create patches
        patches = self.patch(images)
        
        # Create patch embeddings
        embeddings = self.patch_embedding(patches)
        
        # Add class token
        embeddings = self.class_token(embeddings)
        
        # Transformer encoder layers
        for encoder_layer in self.encoder_layers:
            embeddings = encoder_layer(embeddings, training)
        
        # Take the class token output
        class_token_output = embeddings[:, 0]
        
        # MLP head
        logits = self.mlp_head(class_token_output)
        
        return logits
    
    
        



