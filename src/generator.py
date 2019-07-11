import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import os
from collections import OrderedDict
from torch.nn.init import kaiming_normal_
from StyleUtils import *
import torchvision.models as models
from discriminator import StyleDiscriminator
from Model.SE_ResNet_IR import *
device =  'cuda' if torch.cuda.is_available() else 'cpu'
class LayerEpilogue(nn.Module):
    def __init__(self,
                 channels,
                 dlatent_size,
                 use_wscale,
                 use_noise,
                 use_pixel_norm,
                 use_instance_norm,
                 use_styles):
        super(LayerEpilogue, self).__init__()

        if use_noise:
            self.noise = ApplyNoise(channels)
        self.act = nn.LeakyReLU(negative_slope=0.2)

        if use_pixel_norm:
            self.pixel_norm = PixelNorm()
        else:
            self.pixel_norm = None

        if use_instance_norm:
            self.instance_norm = InstanceNorm()
        else:
            self.instance_norm = None

        if use_styles:
            self.style_mod = ApplyStyle(dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None

    def forward(self, x, noise, dlatents_in_slice=None):
        x = self.noise(x, noise)
        x = self.act(x)
        if self.pixel_norm is not None:
            x = self.pixel_norm(x)
        if self.instance_norm is not None:
            #print(x.shape)
            x = self.instance_norm(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)

        return x


class GBlock(nn.Module):
    def __init__(self,
                 res,
                 use_wscale,
                 use_noise,
                 use_pixel_norm,
                 use_instance_norm,
                 noise_input,        # noise
                 dlatent_size=512,   # Disentangled latent (W) dimensionality.
                 use_style=True,     # Enable style inputs?
                 f=None,        # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 factor=2,           # upsample factor.
                 fmap_base=8192,     # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,     # log2 feature map reduction when doubling the resolution.
                 fmap_max=512,       # Maximum number of feature maps in any layer.
                 ):
        super(GBlock, self).__init__()
        self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        # res
        self.res = res

        # blur2d
        self.blur = Blur2d(f)

        # noise
        self.noise_input = noise_input

        if res < 7:
            # upsample method 1
            self.up_sample = Upscale2d(factor)
        else:
            # upsample method 2
            self.up_sample = nn.ConvTranspose2d(self.nf(res-3), self.nf(res-2), 4, stride=2, padding=1)

        # A Composition of LayerEpilogue and Conv2d.
        self.adaIn1 = LayerEpilogue(self.nf(res-2), dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)
        self.conv1  = Conv2d(input_channels=self.nf(res-2), output_channels=self.nf(res-2),
                             kernel_size=3, use_wscale=use_wscale)
        self.adaIn2 = LayerEpilogue(self.nf(res-2), dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)

    def forward(self, x, dlatent):
        x = self.up_sample(x)
        x = self.adaIn1(x, self.noise_input[self.res*2-4], dlatent[:, self.res*2-4])
        x = self.conv1(x)
        x = self.adaIn2(x, self.noise_input[self.res*2-3], dlatent[:, self.res*2-3])
        return x

# =========================================================================
#   Define sub-network
# =========================================================================
class G_mapping(nn.Module):
    def __init__(self,
                 mapping_fmaps=512,
                 dlatent_size=512,
                 resolution=1024,
                 normalize_latents=True,  # Normalize latent vectors (Z) before feeding them to the mapping layers?
                 use_wscale=True,         # Enable equalized learning rate?
                 lrmul=0.01,              # Learning rate multiplier for the mapping layers.
                 gain=2**(0.5)            # original gain in tensorflow.
                 ):
        super(G_mapping, self).__init__()
        self.mapping_fmaps = mapping_fmaps
        self.func = nn.Sequential(
            FC(self.mapping_fmaps, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale)
        )

        self.normalize_latents = normalize_latents
        self.resolution_log2 = int(np.log2(resolution))
        self.num_layers = self.resolution_log2 * 2 - 2
        self.pixel_norm = PixelNorm()
        # - 2 means we start from feature map with height and width equals 4.
        # as this example, we get num_layers = 18.

    def forward(self, x):
        if self.normalize_latents:
            x = self.pixel_norm(x)
        out = self.func(x)
        return out, self.num_layers


class G_synthesis(nn.Module):
    def __init__(self,
                 dlatent_size,                       # Disentangled latent (W) dimensionality.
                 resolution=1024,                    # Output resolution (1024 x 1024 by default).
                 fmap_base=8192,                     # Overall multiplier for the number of feature maps.
                 num_channels=3,                     # Number of output color channels.
                 structure='fixed',                  # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
                 fmap_max=512,                       # Maximum number of feature maps in any layer.
                 fmap_decay=1.0,                     # log2 feature map reduction when doubling the resolution.
                 f=None,                        # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 use_pixel_norm      = False,        # Enable pixelwise feature vector normalization?
                 use_instance_norm   = True,        # Enable instance normalization?
                 use_wscale = True,                  # Enable equalized learning rate?
                 use_noise = True,                   # Enable noise inputs?
                 use_style = True                    # Enable style inputs?
                 ):                             # batch size.
        """
        :param dlatent_size: 512 Disentangled latent(W) dimensionality.
        :param resolution: 1024 x 1024.
        :param fmap_base:
        :param num_channels:
        :param structure: only support 'fixed' mode.
        :param fmap_max:
        """
        super(G_synthesis, self).__init__()

        self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        self.structure = structure
        self.resolution_log2 = int(np.log2(resolution))
        # - 2 means we start from feature map with height and width equals 4.
        # as this example, we get num_layers = 18.
        num_layers = self.resolution_log2 * 2 - 2
        self.num_layers = num_layers

        # Noise inputs.
        self.noise_inputs = []
        for layer_idx in range(num_layers):
            res = layer_idx // 2 + 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noise_inputs.append(torch.randn(*shape).to(device))

        # Blur2d
        self.blur = Blur2d(f)

        # torgb: fixed mode
        self.channel_shrinkage = Conv2d(input_channels=512,
                                        output_channels=512,
                                        kernel_size=3,
                                        use_wscale=use_wscale)
        self.torgb = Conv2d(512, num_channels, kernel_size=1, gain=1, use_wscale=use_wscale)

        # Initial Input Block
        self.const_input = nn.Parameter(torch.ones(1, self.nf(1), 4, 4))
        self.bias = nn.Parameter(torch.ones(self.nf(1)))
        self.adaIn1 = LayerEpilogue(self.nf(1), dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)
        self.conv1  = Conv2d(input_channels=self.nf(1), output_channels=self.nf(1), kernel_size=3, use_wscale=use_wscale)
        self.adaIn2 = LayerEpilogue(self.nf(1), dlatent_size, use_wscale, use_noise, use_pixel_norm,
                                    use_instance_norm, use_style)

        # Common Block
        # 4 x 4 -> 8 x 8
        res = 3
        self.GBlock1 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 8 x 8 -> 16 x 16
        res = 4
        self.GBlock2 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 16 x 16 -> 32 x 32
        res = 5
        self.GBlock3 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 32 x 32 -> 64 x 64
        res = 6
        self.GBlock4 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

    def forward(self, dlatent):
        """
           dlatent: Disentangled latents (W), shape为[minibatch, num_layers, dlatent_size].
        :param dlatent:
        :return:
        """
        images_out = None
        # Fixed structure: simple and efficient, but does not support progressive growing.
        if self.structure == 'fixed':
            # initial block 0:
            x = self.const_input.expand(dlatent.size(0), -1, -1, -1)
            x = x + self.bias.view(1, -1, 1, 1)
            #print(x.shape)
            x = self.adaIn1(x, self.noise_inputs[0], dlatent[:, 0])
            x = self.conv1(x)
            x = self.adaIn2(x, self.noise_inputs[1], dlatent[:, 1])

            # block 1:
            # 4 x 4 -> 8 x 8
            x = self.GBlock1(x, dlatent)

            # block 2:
            # 8 x 8 -> 16 x 16
            x = self.GBlock2(x, dlatent)

            # block 3:
            # 16 x 16 -> 32 x 32
            x = self.GBlock3(x, dlatent)

            # block 4:
            # 32 x 32 -> 64 x 64
            x = self.GBlock4(x, dlatent)
            #print(x.shape)

            x = self.channel_shrinkage(x)
            images_out = self.torgb(x)
            return images_out


class StyleGenerator(nn.Module):
    def __init__(self,
                 mapping_fmaps = 512,               
                 style_mixing_prob=0.9,       # Probability of mixing styles during training. None = disable.
                 truncation_psi=0.7,          # Style strength multiplier for the truncation trick. None = disable.
                 truncation_cutoff=8,          # Number of layers for which to apply the truncation trick. None = disable.
                 **kwargs
                 ):
        super(StyleGenerator, self).__init__()
        self.vgg_all = models.vgg16(pretrained=False)#.features
        self.vgg_all.load_state_dict(torch.load('./vgg16-397923af.pth'))
        self.vgg = self.vgg_all.features
        #self.vgg = SEResNet_IR(50, mode='se_ir')
        #self.vgg.load_state_dict(torch.load('./params_res50IR_cos_CA.pkl'))
        for para in self.vgg.parameters():
            para.requires_grad = False 
        self.mapping_fmaps = mapping_fmaps
        self.style_mixing_prob = style_mixing_prob
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff
        self.input_trans = nn.Linear(1024, self.mapping_fmaps)
        self.mapping = G_mapping(self.mapping_fmaps, **kwargs)
        self.synthesis = G_synthesis(self.mapping_fmaps, **kwargs)

    def forward(self, image):
        
        _ = self.vgg(image)
        for i in range(0,17):
            image = self.vgg[i](image)  # get pool4 size: 32*32  
        shape = image.shape[0]
        image = torch.max(image,1)[0] 
        image = image.view([shape,-1])
        latents1 = self.input_trans(image)
        dlatents1, num_layers = self.mapping(latents1)
        
        #dlatents1 = self.vgg(image)
        num_layers = 18
        # let [N, O] -> [N, num_layers, O]
        # 这里的unsqueeze不能使用inplace操作, 如果这样的话, 反向传播的链条会断掉
        dlatents1 = dlatents1.unsqueeze(1)
        dlatents1 = dlatents1.expand(-1, int(num_layers), -1)

        # Add mixing style mechanism.
        # with torch.no_grad():
        #     latents2 = torch.randn(latents1.shape).to(latents1.device)
        #     dlatents2, num_layers = self.mapping(latents2)
        #     dlatents2 = dlatents2.unsqueeze(1)
        #     dlatents2 = dlatents2.expand(-1, int(num_layers), -1)
        #
        #     # TODO: original NvLABs produce a placeholder "lod", this mechanism was not added here.
        #     cur_layers = num_layers
        #     mix_layers = num_layers
        #     if np.random.random() < self.style_mixing_prob:
        #         mix_layers = np.random.randint(1, cur_layers)
        #
        #     # NvLABs: dlatents = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, tf.shape(dlatents)), dlatents, dlatents2)
        #     for i in range(num_layers):
        #         if i >= mix_layers:
        #             dlatents1[:, i, :] = dlatents2[:, i, :]

        # Apply truncation trick.
        if self.truncation_psi and self.truncation_cutoff:
            coefs = np.ones([1, num_layers, 1], dtype=np.float32)
            for i in range(num_layers):
                if i < self.truncation_cutoff:
                    coefs[:, i, :] *= self.truncation_psi
            dlatents1 = dlatents1 * torch.Tensor(coefs).to(dlatents1.device)

        img = self.synthesis(dlatents1)
        return img
if __name__ == '__main__':
    x = torch.randn(1,3,256,256)
    G = StyleGenerator()
    out = G(x)
    print(out.shape)
    D = StyleDiscriminator()
    z = D(out)
    print(z)


