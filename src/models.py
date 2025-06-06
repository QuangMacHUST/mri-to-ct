import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List

class ResidualBlock(nn.Module):
    """
    Residual Block cho Generator
    """
    def __init__(self, in_features: int):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )
    
    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """
    Generator network cho CycleGAN
    Kiến trúc: Encoder-Decoder với residual blocks
    """
    def __init__(self, input_nc: int = 1, output_nc: int = 1, n_residual_blocks: int = 9):
        super(Generator, self).__init__()
        
        # Encoder (downsampling)
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling layers
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        
        # Decoder (upsampling)
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()  # Output range [-1, 1]
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator
    Phân loại từng patch của ảnh thay vì toàn bộ ảnh
    """
    def __init__(self, input_nc: int = 1, n_layers: int = 3):
        super(PatchGANDiscriminator, self).__init__()
        
        # Không sử dụng InstanceNorm cho layer đầu tiên
        model = [
            nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            model += [
                nn.Conv2d(64 * nf_mult_prev, 64 * nf_mult, 4, stride=2, padding=1),
                nn.InstanceNorm2d(64 * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model += [
            nn.Conv2d(64 * nf_mult_prev, 64 * nf_mult, 4, stride=1, padding=1),
            nn.InstanceNorm2d(64 * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Output layer - không có activation (sẽ dùng loss function để quyết định)
        model += [
            nn.Conv2d(64 * nf_mult, 1, 4, stride=1, padding=1)
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual Loss sử dụng VGG19 network với xử lý đúng input range
    """
    def __init__(self, feature_layers: List[int] = [3, 8, 15, 22], resize: bool = False):
        super(VGGPerceptualLoss, self).__init__()
        
        # Load VGG19 với cách mới để tránh lỗi torchvision::nms
        try:
            # Phương pháp mới (PyTorch 1.13+)
            from torchvision.models import VGG19_Weights
            vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        except ImportError:
            # Fallback cho phiên bản cũ
            try:
                vgg = models.vgg19(pretrained=True)
            except:
                # Nếu vẫn lỗi, tạo model không pretrained
                print("Warning: Không thể load pretrained VGG19, sử dụng random weights")
                vgg = models.vgg19(pretrained=False)
        
        self.features = vgg.features
        
        # Freeze VGG parameters để tiết kiệm memory và tránh update
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.feature_layers = feature_layers
        self.resize = resize
        
        # Register buffers cho ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, x, y):
        """
        Tính perceptual loss giữa x và y
        Input: x, y trong range [-1, 1] (từ Tanh activation)
        """
        # ✅ QUAN TRỌNG: Convert từ [-1,1] về [0,1] trước khi apply ImageNet norm
        x = (x + 1.0) / 2.0
        y = (y + 1.0) / 2.0
        
        # Chuyển từ grayscale sang RGB nếu cần
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.size(1) == 1:
            y = y.repeat(1, 3, 1, 1)
        
        # Resize về 224x224 nếu cần (theo chuẩn ImageNet)
        if self.resize:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Chuẩn hóa theo ImageNet (giờ đã đúng range [0,1])
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        
        loss = 0.0
        x_features = x
        
        # Tối ưu: target không cần gradient
        with torch.no_grad():
            y_features = y
            
        for i, layer in enumerate(self.features):
            x_features = layer(x_features)
            
            # Target features cũng trong no_grad
            with torch.no_grad():
                y_features = layer(y_features)
            
            if i in self.feature_layers:
                # Chỉ x_features cần gradient, y_features đã được detach
                loss += F.mse_loss(x_features, y_features.detach())
        
        return loss


class CycleGAN(nn.Module):
    """
    CycleGAN model hoàn chỉnh với loss weights được điều chỉnh
    """
    def __init__(self, 
                 input_nc: int = 1, 
                 output_nc: int = 1,
                 n_residual_blocks: int = 9,
                 discriminator_layers: int = 3):
        super(CycleGAN, self).__init__()
        
        # Generators
        self.G_MRI2CT = Generator(input_nc, output_nc, n_residual_blocks)
        self.G_CT2MRI = Generator(output_nc, input_nc, n_residual_blocks)
        
        # Discriminators
        self.D_CT = PatchGANDiscriminator(output_nc, discriminator_layers)
        self.D_MRI = PatchGANDiscriminator(input_nc, discriminator_layers)
        
        # Perceptual loss với resize=False để giữ nguyên kích thước ảnh
        self.perceptual_loss = VGGPerceptualLoss(resize=False)
        
        # Loss weights - điều chỉnh cho medical imaging
        self.lambda_cycle = 10.0      # Cycle consistency - quan trọng nhất
        self.lambda_identity = 5.0    # Identity loss - giữ cấu trúc anatomical  
        self.lambda_perceptual = 2.0  # Perceptual loss - tăng để cải thiện visual quality
        self.lambda_adversarial = 1.0 # Adversarial loss - cơ bản
        
    def forward(self, mri, ct):
        """
        Forward pass cho training
        """
        # Generate fake images
        fake_ct = self.G_MRI2CT(mri)
        fake_mri = self.G_CT2MRI(ct)
        
        # Cycle consistency
        rec_mri = self.G_CT2MRI(fake_ct)
        rec_ct = self.G_MRI2CT(fake_mri)
        
        # Identity mapping (giúp bảo toàn cấu trúc anatomical)
        identity_ct = self.G_MRI2CT(ct)
        identity_mri = self.G_CT2MRI(mri)
        
        return {
            'fake_ct': fake_ct,
            'fake_mri': fake_mri,
            'rec_mri': rec_mri,
            'rec_ct': rec_ct,
            'identity_ct': identity_ct,
            'identity_mri': identity_mri
        }
    
    def generator_loss(self, real_mri, real_ct, outputs):
        """
        Tính Generator loss với tất cả thành phần
        """
        fake_ct = outputs['fake_ct']
        fake_mri = outputs['fake_mri'] 
        rec_mri = outputs['rec_mri']
        rec_ct = outputs['rec_ct']
        identity_ct = outputs['identity_ct']
        identity_mri = outputs['identity_mri']
        
        # 1. Adversarial loss - Generator cố gắng đánh lừa Discriminator
        pred_fake_ct = self.D_CT(fake_ct)
        pred_fake_mri = self.D_MRI(fake_mri)
        
        # Sử dụng MSE loss thay vì BCE cho LSGAN (ổn định hơn)
        loss_gan_ct = F.mse_loss(pred_fake_ct, torch.ones_like(pred_fake_ct))
        loss_gan_mri = F.mse_loss(pred_fake_mri, torch.ones_like(pred_fake_mri))
        loss_gan = (loss_gan_ct + loss_gan_mri) * 0.5
        
        # 2. Cycle consistency loss - F(G(x)) ≈ x và G(F(y)) ≈ y
        loss_cycle_mri = F.l1_loss(rec_mri, real_mri)
        loss_cycle_ct = F.l1_loss(rec_ct, real_ct)
        loss_cycle = (loss_cycle_mri + loss_cycle_ct) * 0.5
        
        # 3. Identity loss - G(y) ≈ y và F(x) ≈ x (giữ cấu trúc khi input đã đúng domain)
        loss_identity_ct = F.l1_loss(identity_ct, real_ct)
        loss_identity_mri = F.l1_loss(identity_mri, real_mri)
        loss_identity = (loss_identity_ct + loss_identity_mri) * 0.5
        
        # 4. Perceptual loss - chỉ áp dụng cho direction chính (MRI->CT)
        loss_perceptual = self.perceptual_loss(fake_ct, real_ct)
        
        # Total generator loss với trọng số đã điều chỉnh
        total_loss = (self.lambda_adversarial * loss_gan + 
                     self.lambda_cycle * loss_cycle +
                     self.lambda_identity * loss_identity +
                     self.lambda_perceptual * loss_perceptual)
        
        return {
            'total': total_loss,
            'gan': loss_gan,
            'cycle': loss_cycle,
            'identity': loss_identity,
            'perceptual': loss_perceptual
        }
    
    def discriminator_loss(self, real_images, fake_images, discriminator):
        """
        Tính Discriminator loss (LSGAN loss cho ổn định)
        """
        # Real images - target = 1
        pred_real = discriminator(real_images)
        loss_real = F.mse_loss(pred_real, torch.ones_like(pred_real))
        
        # Fake images - target = 0, quan trọng: detach để không backprop vào Generator
        pred_fake = discriminator(fake_images.detach())
        loss_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))
        
        # Total discriminator loss
        total_loss = (loss_real + loss_fake) * 0.5
        
        return total_loss


def weights_init_normal(m):
    """
    Khởi tạo trọng số cho model theo chuẩn CycleGAN
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        # Khởi tạo convolution layers
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        # Khởi tạo batch normalization layers  
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("InstanceNorm2d") != -1:
        # Instance normalization thường không cần khởi tạo đặc biệt
        if hasattr(m, "weight") and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0) 