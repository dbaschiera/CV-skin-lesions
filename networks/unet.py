import tensorflow as tf
from tensorflow.keras import layers, models

class UNet(tf.keras.Model):
    def __init__(self, input_size=(256, 256, 3), num_classes=1):
        super(UNet, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Contracting Path (Encoder)
        self.enc_conv1 = self.conv_block(64)
        self.enc_conv2 = self.conv_block(128)
        self.enc_conv3 = self.conv_block(256)
        self.enc_conv4 = self.conv_block(512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(1024)
        
        # Expansive Path (Decoder)
        self.upconv4 = self.upconv_block(512)
        self.upconv3 = self.upconv_block(256)
        self.upconv2 = self.upconv_block(128)
        self.upconv1 = self.upconv_block(64)
        
        # Output layer with a 1x1 convolution (sigmoid for binary classification)
        self.output_layer = layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid', padding='same')

    def conv_block(self, filters, kernel_size=(3, 3), padding='same', strides=(1, 1)):
        block = models.Sequential([
            layers.Conv2D(filters, kernel_size, strides=strides, padding=padding),
            layers.ReLU(),
            layers.Conv2D(filters, kernel_size, strides=strides, padding=padding),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2))  # Max-pooling with size (2, 2)
        ])
        return block

    def upconv_block(self, filters, kernel_size=(3, 3), padding='same', strides=(1, 1)):
        block = models.Sequential([
            layers.Conv2DTranspose(filters, kernel_size, strides=(2, 2), padding=padding),  # Upsample with 2x2 strides
            layers.ReLU(),
            layers.Conv2D(filters, kernel_size, strides=strides, padding=padding),
            layers.ReLU()
        ])
        return block
    
    def call(self, inputs):
        # Contracting path (encoder)
        enc1 = self.enc_conv1(inputs)
        enc2 = self.enc_conv2(enc1)
        enc3 = self.enc_conv3(enc2)
        enc4 = self.enc_conv4(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4)
        
        # Expansive path (decoder)
        up4 = self.upconv4(bottleneck)
        up3 = self.upconv3(up4)
        up2 = self.upconv2(up3)
        up1 = self.upconv1(up2)
        
        # Output layer (1x1 convolution to match original image size)
        output = self.output_layer(up1)
        
        # Resize to ensure it matches the input size (256x256)
        output = tf.image.resize(output, size=(256, 256))  # Resize output to 256x256 if needed
        
        return output
