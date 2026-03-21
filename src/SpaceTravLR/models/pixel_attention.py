import pyro
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
import torch.nn as nn
import numpy as np
import torch    
import functools
from torch.nn.utils.parametrizations import weight_norm
from sklearn.linear_model import ARDRegression

from SpaceTravLR.models.vit_blocks import ViTBlock, get_positional_embeddings, patchify

pyro.clear_param_store()

device = torch.device(
    "mps" if torch.backends.mps.is_available() 
    else "cuda" if torch.cuda.is_available() 
    else "cpu"
)

use_conditional_conv = device == 'gpu'

class _cluster_routing(nn.Module):

    def __init__(self, num_clusters, pool_emb, num_experts, dropout_rate=0.1):
        super(_cluster_routing, self).__init__()
        
        self.dropout = nn.Dropout(dropout_rate)
        self.cluster_emb = nn.Embedding(num_clusters, pool_emb)
        self.fc = nn.Linear(pool_emb*2, num_experts)

    def forward(self, spatial_f, labels):
        spatial_f = spatial_f.flatten(1)
        emb = self.cluster_emb(labels)
        x = self.fc(torch.cat([spatial_f, emb], dim=1))
        x = self.dropout(x)
        return F.sigmoid(x)

    # def forward_(self, spatial_f, labels):
    #     spatial_f = spatial_f.flatten()
    #     emb = self.cluster_emb(labels)
    #     x = self.fc(torch.cat([spatial_f, emb]))
    #     x = self.dropout(x)
    #     return F.sigmoid(x)
    
class ConditionalConv2D(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', num_experts=5, dropout_rate=0.1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(ConditionalConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(2, 2))
        pool_emb = torch.mul(*self._avg_pooling.keywords['output_size']) * in_channels
        
        self._routing_fn = _cluster_routing(
            num_clusters=in_channels,
            pool_emb=pool_emb,
            num_experts=num_experts, 
            dropout_rate=dropout_rate
        )

        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))
        
        self.reset_parameters()

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    

    def forward(self, inputs, input_labels):
        res = []
        
        assert inputs.shape[0] == input_labels.shape[0]
        pooled_inputs = self._avg_pooling(inputs)
        routing_weights = self._routing_fn(pooled_inputs, input_labels)
        
        # Get the index of the highest weight for each input
        max_weight_indices = torch.argmax(routing_weights, dim=1)
        
        # Select the kernel with the highest weight for each input
        kernels = self.weight[max_weight_indices]
        for inputx, kernel in zip(inputs, kernels):
            out = self._conv_forward(inputx.unsqueeze(0), kernel)
            res.append(out)


        # for inputx, label in zip(inputs, input_labels):
        #     inputx = inputx.unsqueeze(0)
        #     pooled_inputs = self._avg_pooling(inputx)
        #     routing_weights = self._routing_fn(pooled_inputs, label)
        #     kernels = torch.sum(routing_weights[:, None, None, None, None] * self.weight, 0)
        #     out = self._conv_forward(inputx, kernels)
        #     res.append(out)
        
        return torch.cat(res, dim=0)
    
class NicheAttentionNetwork(nn.Module):
     
    def __init__(self, n_regulators, in_channels, spatial_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.spatial_dim = spatial_dim
        self.dim = n_regulators+1
        
        # if use_conditional_conv:
        #     self.conditional_conv = ConditionalConv2D(
        #         self.in_channels, self.in_channels, 1, num_experts=self.in_channels)
        
        self.conditional_conv = nn.Conv2d(self.in_channels, self.in_channels, 1)

        self.sigmoid = nn.Sigmoid()

        self.conv_layers = nn.Sequential(
            weight_norm(nn.Conv2d(in_channels, 32, kernel_size=3, padding='same')),
            nn.PReLU(init=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            weight_norm(nn.Conv2d(32, 64, kernel_size=3, padding='same')),
            nn.PReLU(init=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            weight_norm(nn.Conv2d(64, 128, kernel_size=3, padding='same')),
            nn.PReLU(init=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # self.cluster_emb = nn.Embedding(self.in_channels, 128)

        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.PReLU(init=0.1),
            nn.Linear(64, self.dim)
        )

        # self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.output_activation = nn.Tanh()


    def forward(self, spatial_maps, cluster_info):
        # att = self.sigmoid(self.conditional_conv(spatial_maps, cluster_info))
        att = self.sigmoid(self.conditional_conv(spatial_maps))
        out = att * spatial_maps
        out = self.conv_layers(out)
        # emb = self.cluster_emb(cluster_info) * self.alpha
        # out = out + emb 

        betas = self.mlp(out)
        betas = self.output_activation(betas)

        return betas
    
##Live Model
class CellularNicheNetwork(nn.Module):

    @staticmethod
    def make_vision_model(input_channels=1, out_dim=64, kernel_size=3):

        return nn.Sequential(
            weight_norm(nn.Conv2d(input_channels, 16, kernel_size=kernel_size, padding='same')),
            nn.BatchNorm2d(16),
            nn.PReLU(init=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            weight_norm(nn.Conv2d(16, 32, kernel_size=kernel_size, padding='same')),
            nn.BatchNorm2d(32),
            nn.PReLU(init=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            weight_norm(nn.Conv2d(32, out_dim, kernel_size=kernel_size, padding='same')),
            nn.BatchNorm2d(out_dim),
            nn.PReLU(init=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
    
    @classmethod
    def from_pretrained(cls, trained_model, n_modulators, anchors=None, spatial_dim=64, n_clusters=7):
        cnn = cls.make_vision_model()
        cnn.load_state_dict(trained_model.conv_layers.state_dict())
        model = cls(n_modulators, anchors, spatial_dim, n_clusters)
        model.conv_layers = cnn
        return model

     
    def __init__(self, n_modulators, anchors=None, spatial_dim=64, n_clusters=7):
        super().__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.spatial_dim = spatial_dim
        self.dim = n_modulators+1
        if anchors is None:
            anchors = np.ones(self.dim)

        self.anchors = torch.from_numpy(anchors).float().to(device)

        # self.anchors = torch.nn.Parameter(self.anchors, requires_grad=True)
        # self.conditional_conv = nn.Conv2d(self.in_channels, self.in_channels, 1)
        # self.sigmoid = nn.Sigmoid()

        self.conv_layers = self.make_vision_model(input_channels=self.in_channels)

        self.spatial_features_mlp = nn.Sequential(
            nn.Linear(n_clusters, 16),
            nn.PReLU(init=0.1),
            nn.Linear(16, 32),
            nn.PReLU(init=0.1),
            nn.Linear(32, 64)
        )

        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.PReLU(init=0.1),
            nn.Linear(64, self.dim)
        )

        # self.output_activation = nn.Tanh()
        self.output_activation = nn.Sigmoid()
        # self.output_activation = nn.GELU()
        # self.output_activation = nn.Identity()
        # self.output_activation = nn.Softplus()


    def get_betas(self, spatial_maps, spatial_features):
        out = self.conv_layers(spatial_maps)
        sp_out = self.spatial_features_mlp(spatial_features)
        out = out+sp_out
        betas = self.mlp(out)
        betas = self.output_activation(betas)

        return betas*self.anchors
    
    @staticmethod
    def predict_y(inputs_x, betas):
        return torch.matmul(
                inputs_x.unsqueeze(1), 
                betas[:, 1:].unsqueeze(2)
            ).squeeze(1).squeeze(1) + \
                betas[:, 0]
    
    def forward(self, spatial_maps, inputs_x, spatial_features):
        betas = self.get_betas(spatial_maps, spatial_features)
        y_pred = self.predict_y(inputs_x, betas)
        
        return y_pred
    
class CellularViT(nn.Module):
    
    @classmethod
    def from_pretrained(cls, trained_model, n_modulators, anchors=None, spatial_dim=64, n_clusters=7):
        cnn = cls.make_vision_model()
        cnn.load_state_dict(trained_model.conv_layers.state_dict())
        model = cls(n_modulators, anchors, spatial_dim, n_clusters)
        model.conv_layers = cnn
        return model

     
    def __init__(self, n_modulators, anchors=None, spatial_dim=64, n_clusters=7, n_patches=4, n_blocks=4, hidden_d=16, n_heads=8):
        super().__init__()
        in_channels = 1
        self.out_channels = 1
        self.spatial_dim = spatial_dim
        self.dim = n_modulators+1
        if anchors is None:
            anchors = np.ones(self.dim)

        self.anchors = torch.from_numpy(anchors).float().to(device)
        
        self.dim = n_modulators+1
        if anchors is None:
            anchors = np.ones(self.dim)

        self.anchors = torch.from_numpy(anchors).float().to(device)
        
        self.in_channels = in_channels
        self.spatial_dim = spatial_dim
        
        chw = (in_channels, spatial_dim, spatial_dim) # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        
        # Input and patches sizes
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))       # Consider removing
        self.pos_embed = nn.Parameter(get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_d))
        self.pos_embed.requires_grad = False

        self.blocks = nn.ModuleList(
            [ViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])


        self.spatial_features_mlp = nn.Sequential(
            nn.Linear(n_clusters, 16),
            nn.PReLU(init=0.1),
            nn.Linear(16, 32),
            nn.PReLU(init=0.1),
            nn.Linear(32, self.hidden_d)
        )


        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(16, self.dim)
        )

        self.output_activation = nn.Sigmoid()
    
    
    def get_betas(self, spatial_maps, spatial_features):
        n, c, h, w = spatial_maps.shape 
        patches = patchify(spatial_maps, self.n_patches).to(self.pos_embed.device)
        sp_out = self.spatial_features_mlp(spatial_features)
        tokens = self.linear_mapper(patches) 
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        out = tokens + self.pos_embed.repeat(n, 1, 1)
        
        for j, block in enumerate(self.blocks):
            out = block(out)
            
            
        betas = self.mlp(out[:, 0]+sp_out)
        betas = self.output_activation(betas) * 1.5
        
        return betas*self.anchors
    
    @staticmethod
    def predict_y(inputs_x, betas):
        return torch.matmul(
                inputs_x.unsqueeze(1), 
                betas[:, 1:].unsqueeze(2)
            ).squeeze(1).squeeze(1) + \
                betas[:, 0]
                
    
    def forward(self, spatial_maps, inputs_x, spatial_features):
        betas = self.get_betas(spatial_maps, spatial_features)
        y_pred = self.predict_y(inputs_x, betas)
        
        return y_pred