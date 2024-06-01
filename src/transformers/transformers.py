import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np



# Define the Patch layer
class Patch(layers.Layer):
    """
    This class for make images into patches

    Args:
        patch_size (int)

    Return:
        patches
    """
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

# Define the PatchEmbedding layer
class PatchEmbedding(layers.Layer):
    """
    This class for represent patches as Embeddings with position embedding

    Args:
        num_patches (int)
        embedding_dim (int)

    Returns:
        embedded patches
    """
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

# Define the ClassToken layer
class ClassToken(layers.Layer):
    """
    this class for add class token for aggregate information fro, all patches for classification
    """
    def __init__(self):
        super(ClassToken, self).__init__()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable=True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]
        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls

# Define the MLP layer
class MLP(layers.Layer):
    """
    This class creates simple Multi-Layer Perceptron

    Args:
        hidden_units (int)
        dropout_rate (float)

    Returns:
        x (array)
    """
    def __init__(self, hidden_units, dropout_rate):
        super(MLP, self).__init__()
        self.dense_layers = [layers.Dense(units, activation=tf.nn.gelu) for units in hidden_units]
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x):
        for dense in self.dense_layers:
            x = dense(x)
        return self.dropout(x)

# Define the TransformerEncoderLayer
class TransformerEncoderLayer(layers.Layer):
    """
    This class for Encodes and processes input embeddings using self-attention and feedforward networks.

    Args:
        num_heads (int) 
        hidden_dim (int) 
        mlp_dim (int) 
        dropout_rate (float)

    Returns:
        x (array)
    """
    def __init__(self, num_heads, hidden_dim, mlp_dim, dropout_rate):
        super(TransformerEncoderLayer, self).__init__()
        self.ln1 = layers.LayerNormalization()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim)
        self.add1 = layers.Add()
        self.ln2 = layers.LayerNormalization()
        self.mlp = MLP([mlp_dim, hidden_dim], dropout_rate)
        self.add2 = layers.Add()

    def call(self, x, training=False):
        skip_1 = x
        x = self.ln1(x)
        x = self.mha(x, x)
        x = self.add1([x, skip_1])

        skip_2 = x
        x = self.ln2(x)
        x = self.mlp(x, training=training)
        x = self.add2([x, skip_2])

        return x

# Define the ViTModel
class ViTModel(tf.keras.Model):
    """
    This class  for Integrates all components to create the complete Vision Transformer (ViT) model.

    Args:
        image_size (int) 
        patch_size  
        num_layers
        num_classes
        hidden_dim
        num_heads
        mlp_dim
        dropout_rate

    Return:
    """
    def __init__(self, image_size, patch_size, num_layers, num_classes, hidden_dim, num_heads, mlp_dim, dropout_rate=0.1):
        super(ViTModel, self).__init__()
        num_patches = (image_size // patch_size) ** 2

        # Initialize the components
        self.patch_extractor = Patch(patch_size)
        self.patch_embedding = PatchEmbedding(num_patches=num_patches, embedding_dim=hidden_dim)
        self.class_token = ClassToken()
        self.transformer_encoder_layers = [
            TransformerEncoderLayer(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout_rate,
            ) for _ in range(num_layers)
        ]
        self.ln = layers.LayerNormalization()
        self.mlp_head = layers.Dense(num_classes)

    def call(self, images, training=False):
        # Extract patches from the image
        patches = self.patch_extractor(images)
        
        # Embed the patches
        x = self.patch_embedding(patches)

        # Add the class token
        cls_token = self.class_token(x)
        x = tf.concat([cls_token, x], axis=1)

        # Pass through the transformer encoder layers
        for layer in self.transformer_encoder_layers:
            x = layer(x, training=training)

        # Normalize
        x = self.ln(x)

        # Use the class token's representation for classification
        cls_token_final = x[:, 0]
        logits = self.mlp_head(cls_token_final)

        return logits
    


    
class SimpleCNN(tf.keras.Model):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Define the layers
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu')
        self.pool1 = layers.MaxPooling2D((2, 2))
        
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = layers.MaxPooling2D((2, 2))
        
        self.conv3 = layers.Conv2D(128, (3, 3), activation='relu')
        self.pool3 = layers.MaxPooling2D((2, 2))

        self.conv4 = layers.Conv2D(256, (3, 3), activation='relu')
        self.pool4 = layers.MaxPooling2D((2, 2))
        

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.pool4(x)
        
        return x


class ConViT(Model):
    def __init__(self, image_size, patch_size, num_layers, num_classes, d_model, num_heads, mlp_dim, dropout_rate):
        super(ViTModel, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.patch = 0
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
    
    
        



