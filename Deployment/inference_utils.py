# inference_utils.py
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision.models import vgg16_bn
import segmentation_models_pytorch as smp

# SegNet definition (matching your training model)
class SegNet(nn.Module):
    def __init__(self, num_classes=1, in_channels=3):
        super(SegNet, self).__init__()
        # Do not request pretrained weights at runtime to avoid network download.
        # The model will be fully initialized from the provided checkpoint.
        vgg = vgg16_bn(weights=None)
        features = list(vgg.features.children())
        self.enc1 = nn.Sequential(*features[0:6])
        self.enc2 = nn.Sequential(*features[6:13])
        self.enc3 = nn.Sequential(*features[13:23])
        self.enc4 = nn.Sequential(*features[23:33])
        self.enc5 = nn.Sequential(*features[33:43])
        self.maxpool5 = features[43]
        self.dec5 = self._decoder_block(512, 512)
        self.dec4 = self._decoder_block(512, 256)
        self.dec3 = self._decoder_block(256, 128)
        self.dec2 = self._decoder_block(128, 64)
        self.dec1 = self._decoder_block(64, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.activation = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x5 = self.maxpool5(x5)
        d5 = self.dec5(x5)
        d4 = self.dec4(d5)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        out = self.final_conv(d1)
        return self.activation(out)

# preprocessing helper using smp encoder fn (same as training)
def get_preprocessing(encoder_name='vgg16', encoder_weights='imagenet'):
    """
    Get preprocessing function for the specified encoder.
    
    Args:
        encoder_name: 'vgg16' for SegNet or 'resnet50' for DeepLabV3Plus
        encoder_weights: 'imagenet' or None
    """
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, encoder_weights)
    # preprocessing_fn expects images in uint8 and returns normalized float32
    def preprocess_image(image: np.ndarray, target_size=512):
        # image: H, W, C RGB uint8 (0-255)
        # resize and pad to target_size (same strategy as training)
        h, w = image.shape[:2]
        scale = 1.0
        if max(h, w) > target_size:
            scale = target_size / max(h, w)
            image = cv2.resize(image, (int(w*scale), int(h*scale)))
            h, w = image.shape[:2]
        pad_bottom = target_size - h
        pad_right = target_size - w
        image = cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0)
        image = preprocessing_fn(image)  # normalizes & returns float32 HWC
        image = np.transpose(image, (2, 0, 1)).astype('float32')  # CHW
        return image, scale
    return preprocess_image

# load model helper
def load_model(weights_path: str, model_type='segnet', device='cpu'):
    """
    Load model from weights file.
    
    Args:
        weights_path: Path to the .pth file
        model_type: 'segnet' or 'deeplabv3plus'
        device: 'cpu' or 'cuda'
    """
    if model_type.lower() == 'segnet':
        model = SegNet(num_classes=1, in_channels=3)
    elif model_type.lower() == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name='resnet50',
            encoder_weights=None,  # We'll load from checkpoint
            in_channels=3,
            classes=1,
            activation='sigmoid'
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'segnet' or 'deeplabv3plus'")
    
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# postprocess mask
def postprocess_mask(prediction: np.ndarray, orig_shape, scale):
    # prediction: (1, H, W) float in [0,1] after sigmoid threshold applied elsewhere
    mask = (prediction > 0.5).astype(np.uint8)[0]  # H, W
    # crop to unpadded scaled original if scale used
    h_orig = int(orig_shape[0] * scale)
    w_orig = int(orig_shape[1] * scale)
    mask = mask[:h_orig, :w_orig]
    if scale != 1.0:
        mask = cv2.resize(mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask
