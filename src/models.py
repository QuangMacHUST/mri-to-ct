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
        
        # Downsampling layers - BATCH_SIZE=7 OPTIMIZED
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_features),  # Changed: InstanceNorm2d → BatchNorm2d for batch_size=7
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        
        # Decoder (upsampling) - BATCH_SIZE=7 OPTIMIZED
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_features),  # Changed: InstanceNorm2d → BatchNorm2d for batch_size=7
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
                nn.BatchNorm2d(64 * nf_mult),  # Changed: InstanceNorm2d → BatchNorm2d for batch_size=7
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model += [
            nn.Conv2d(64 * nf_mult_prev, 64 * nf_mult, 4, stride=1, padding=1),
            nn.BatchNorm2d(64 * nf_mult),  # Changed: InstanceNorm2d → BatchNorm2d for batch_size=7
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
        Tính perceptual loss giữa x và y với enhanced scaling cho medical imaging
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
                # ENHANCED: Scale feature loss để có magnitude phù hợp với medical imaging
                feature_loss = F.mse_loss(x_features, y_features.detach())
                
                # MEDICAL IMAGING BOOST: Scale theo layer depth để cân bằng contribution
                if i <= 8:      # Early layers (texture, edges)
                    scale_factor = 100.0  # Boost low-level features 
                elif i <= 15:   # Mid layers (patterns)
                    scale_factor = 50.0   # Moderate boost
                else:           # High layers (semantics) 
                    scale_factor = 25.0   # Conservative boost
                
                loss += feature_loss * scale_factor
        
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
        
        # Enhanced SSIM calculator cho medical imaging - FIX PLATEAU 0.8
        try:
            from torchmetrics import StructuralSimilarityIndexMeasure
            # ULTIMATE FIX: Optimized parameters dựa trên medical imaging research
            # kernel_size=5 (reduced từ 7), sigma=0.8 (reduced từ 1.0) cho less blur
            self.ssim_calc = StructuralSimilarityIndexMeasure(
                data_range=None,           # Auto-detect range - KEY FIX!
                kernel_size=5,             # OPTIMIZED: 7→5 cho medical patches
                gaussian_kernel=True,
                sigma=0.8,                # OPTIMIZED: 1.0→0.8 reduced blur  
                reduction='elementwise_mean',
                k1=0.01, k2=0.03         # SSIM stability constants
            ).cuda()
            print("✅ ULTIMATE SSIM calculator khởi tạo với breakthrough configuration")
        except Exception as e:
            print(f"⚠️ Warning: Enhanced SSIM failed, using fallback: {e}")
            # Fallback với basic config nhưng vẫn dùng data_range=None
            try:
                self.ssim_calc = StructuralSimilarityIndexMeasure(data_range=None).cuda()
            except:
                self.ssim_calc = None
                print("⚠️ Warning: SSIM calculator không khả dụng, sẽ fallback về L1-only")
        
        # OPTIMIZED loss weights để đạt tỷ lệ lý tưởng cho Medical Imaging  
        # Target Ratio: Cycle 50-55%, Adversarial 25-30%, Perceptual 15-25%
        self.lambda_cycle = 6.0       # MODERATE: 5.0 → 6.0 để đạt 50-55% contribution
        self.lambda_identity = 0.0    # Disabled for cross-modal medical imaging
        self.lambda_perceptual = 40.0 # BOOST: 25.0 → 40.0 để đạt 15-25% contribution (compensate cho base value thấp)
        self.lambda_adversarial = 1.0 # STANDARD: Giữ nguyên làm reference baseline
        
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
    
        # ✅ IDENTITY LOSS HOÀN TOÀN DISABLED CHO Y TẾ
        # Trong medical imaging: MRI và CT là hai modality khác nhau hoàn toàn
        # Identity mapping không có ý nghĩa khi input và output domains khác biệt cơ bản
        # => Không tính identity loss
        
        return {
            'fake_ct': fake_ct,
            'fake_mri': fake_mri,
            'rec_mri': rec_mri,
            'rec_ct': rec_ct,
            'identity_ct': None,  # Disabled
            'identity_mri': None  # Disabled
        }
    
    def generator_loss(self, real_mri, real_ct, outputs):
        """
        Tính Generator loss với tất cả thành phần
        """
        fake_ct = outputs['fake_ct']
        fake_mri = outputs['fake_mri'] 
        rec_mri = outputs['rec_mri']
        rec_ct = outputs['rec_ct']
        
        # 1. Adversarial loss - Generator cố gắng đánh lừa Discriminator
        pred_fake_ct = self.D_CT(fake_ct)
        pred_fake_mri = self.D_MRI(fake_mri)
        
        # Sử dụng MSE loss thay vì BCE cho LSGAN (ổn định hơn)
        loss_gan_ct = F.mse_loss(pred_fake_ct, torch.ones_like(pred_fake_ct))
        loss_gan_mri = F.mse_loss(pred_fake_mri, torch.ones_like(pred_fake_mri))
        loss_gan = (loss_gan_ct + loss_gan_mri) * 0.5
        
        # 2. Enhanced Cycle Consistency Loss với SSIM + Gradient
        # Theo nghiên cứu: L1 + SSIM + Gradient cho visual quality tốt hơn
        loss_cycle_l1_mri = F.l1_loss(rec_mri, real_mri)
        loss_cycle_l1_ct = F.l1_loss(rec_ct, real_ct)
        
        # Thêm SSIM loss cho cycle consistency (key improvement!)
        if self.ssim_calc is not None:
            try:
                # Sử dụng SSIM calculator đã khởi tạo trong __init__
                # Đảm bảo tensor có đúng device
                rec_mri_cuda = rec_mri.to(self.ssim_calc.device)
                real_mri_cuda = real_mri.to(self.ssim_calc.device)
                rec_ct_cuda = rec_ct.to(self.ssim_calc.device)
                real_ct_cuda = real_ct.to(self.ssim_calc.device)
                
                # Convert từ [-1,1] về [0,1] cho SSIM calculation
                rec_mri_01 = (rec_mri_cuda + 1.0) / 2.0
                real_mri_01 = (real_mri_cuda + 1.0) / 2.0
                rec_ct_01 = (rec_ct_cuda + 1.0) / 2.0
                real_ct_01 = (real_ct_cuda + 1.0) / 2.0
                
                # Clamp để đảm bảo trong [0,1] range
                rec_mri_01 = torch.clamp(rec_mri_01, 0.0, 1.0)
                real_mri_01 = torch.clamp(real_mri_01, 0.0, 1.0)
                rec_ct_01 = torch.clamp(rec_ct_01, 0.0, 1.0)
                real_ct_01 = torch.clamp(real_ct_01, 0.0, 1.0)
                
                # Tính SSIM score (0-1, 1 = perfect similarity)
                ssim_mri = self.ssim_calc(rec_mri_01, real_mri_01)
                ssim_ct = self.ssim_calc(rec_ct_01, real_ct_01)
                
                # SSIM loss = 1 - SSIM (0 = perfect, 1 = worst)
                loss_cycle_ssim = (1.0 - ssim_mri) + (1.0 - ssim_ct)
                
                # Gradient loss để preserve edges (quan trọng cho medical imaging)
                def gradient_loss(pred, target):
                    # Sobel operators để tính gradient
                    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
                    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
                    
                    # Tính gradient cho prediction và target
                    pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
                    pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
                    target_grad_x = F.conv2d(target, sobel_x, padding=1)
                    target_grad_y = F.conv2d(target, sobel_y, padding=1)
                    
                    # Gradient magnitude
                    pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
                    target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
                    
                    return F.l1_loss(pred_grad, target_grad)
                
                # Tính gradient loss cho cả MRI và CT
                loss_grad_mri = gradient_loss(rec_mri, real_mri)
                loss_grad_ct = gradient_loss(rec_ct, real_ct)
                loss_cycle_gradient = (loss_grad_mri + loss_grad_ct) * 0.5
                
                # Combined enhanced cycle loss: L1 + α*SSIM + β*Gradient
                # Theo nghiên cứu breakthrough SSIM strategy
                alpha_ssim = 0.4      # Tăng weight cho SSIM để breakthrough
                beta_gradient = 0.2   # Weight cho gradient preservation
                
                loss_cycle = ((loss_cycle_l1_mri + loss_cycle_l1_ct) * 0.5 + 
                             alpha_ssim * loss_cycle_ssim +
                             beta_gradient * loss_cycle_gradient)
                             
            except Exception as e:
                # Fallback to L1 only if SSIM fails
                print(f"⚠️ SSIM calculation failed: {e}")
                loss_cycle = (loss_cycle_l1_mri + loss_cycle_l1_ct) * 0.5
        else:
            # Fallback to L1 only if SSIM calculator không khả dụng
            loss_cycle = (loss_cycle_l1_mri + loss_cycle_l1_ct) * 0.5
        
        # 3. Identity loss - HOÀN TOÀN LOẠI BỎ cho medical imaging
        loss_identity = torch.tensor(0.0, device=real_mri.device, requires_grad=True)
        
        # 4. Enhanced Perceptual loss - áp dụng cho CẢ HAI directions
        # Tính perceptual loss cho cả MRI->CT và CT->MRI để enhance visual quality
        loss_perceptual_mri2ct = self.perceptual_loss(fake_ct, real_ct)     # MRI→CT direction  
        loss_perceptual_ct2mri = self.perceptual_loss(fake_mri, real_mri)   # CT→MRI direction

        # Combined perceptual loss với weight cho medical imaging priority
        # MRI→CT là primary task nên weight cao hơn
        loss_perceptual = (0.7 * loss_perceptual_mri2ct + 0.3 * loss_perceptual_ct2mri)
        
        # Total generator loss - KHÔNG BAO GỒM identity loss
        total_loss = (self.lambda_adversarial * loss_gan + 
                     self.lambda_cycle * loss_cycle +
                     self.lambda_perceptual * loss_perceptual)
        
        return {
            'total': total_loss,
            'gan': loss_gan,
            'cycle': loss_cycle,
            'identity': loss_identity,  # Luôn = 0
            'perceptual': loss_perceptual
        }
    
    def discriminator_loss(self, real_mri, real_ct, outputs):
        """
        Tính loss cho cả 2 discriminators (CT và MRI)
        """
        fake_ct = outputs['fake_ct']
        fake_mri = outputs['fake_mri']
        
        # Discriminator CT loss
        loss_D_CT = self._single_discriminator_loss(real_ct, fake_ct, self.D_CT)
        
        # Discriminator MRI loss  
        loss_D_MRI = self._single_discriminator_loss(real_mri, fake_mri, self.D_MRI)
        
        # Total discriminator loss
        total_loss = (loss_D_CT + loss_D_MRI) * 0.5
        
        return {
            'total': total_loss,
            'D_CT': loss_D_CT,
            'D_MRI': loss_D_MRI
        }
    
    def _single_discriminator_loss(self, real_images, fake_images, discriminator):
        """
        Tính Discriminator loss cho một discriminator với RESEARCH-BASED scaling
        LSGAN loss cho ổn định + 0.5x scaling để balance G-D training
        """
        # Real images - target = 1
        pred_real = discriminator(real_images)
        loss_real = F.mse_loss(pred_real, torch.ones_like(pred_real))
        
        # Fake images - target = 0, quan trọng: detach để không backprop vào Generator
        pred_fake = discriminator(fake_images.detach())
        loss_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))
        
        # RESEARCH-BASED: 0.5x scaling để balance G-D training ratio
        # Original CycleGAN paper: D loss được halved để tránh D dominance
        total_loss = 0.5 * (loss_real + loss_fake)
        
        return total_loss

    def update_loss_weights(self, lambda_cycle=None, lambda_identity=None, 
                           lambda_perceptual=None, lambda_adversarial=None):
        """
        Dynamically update loss weights during training
        Hữu ích khi cần fine-tune loss balance
        """
        if lambda_cycle is not None:
            self.lambda_cycle = lambda_cycle
        if lambda_identity is not None:
            self.lambda_identity = lambda_identity
        if lambda_perceptual is not None:
            self.lambda_perceptual = lambda_perceptual
        if lambda_adversarial is not None:
            self.lambda_adversarial = lambda_adversarial
            
        print(f"🔄 Updated loss weights:")
        print(f"   Cycle: {self.lambda_cycle}")
        print(f"   Identity: {self.lambda_identity}")
        print(f"   Perceptual: {self.lambda_perceptual}")
        print(f"   Adversarial: {self.lambda_adversarial}")
    
    def get_loss_weights_info(self):
        """Return current loss weights as dict"""
        return {
            'lambda_cycle': self.lambda_cycle,
            'lambda_identity': self.lambda_identity,
            'lambda_perceptual': self.lambda_perceptual,
            'lambda_adversarial': self.lambda_adversarial
        }


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