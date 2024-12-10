from functools import partial
from timm.models.layers import trunc_normal_, lecun_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
from Channel_allocation import Dydim_S, Dydim_M, Dydim_L
from gcn_lib import *

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Node Embedding
    """
    def __init__(self, img_size=224, kernel_size=16,  stride=16, padding=0, in_chans=3, embed_dim=768):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj(x)
        x = self.norm(x)
        x = x.permute(0,2,3,1)
        return x



class FirstPatchEmbed(nn.Module):
    """ 2D Image to Node Embedding
    """
    def __init__(self, kernel_size=3,  stride=2, padding=1, in_chans=3, embed_dim=768):
        super().__init__()
        
        self.proj1 = nn.Conv2d(in_chans, embed_dim//2, kernel_size=kernel_size, stride=stride, padding=padding )
        self.norm1 = nn.BatchNorm2d(embed_dim // 2)
        self.act = nn.GELU()
        self.proj2 = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding )
        self.norm2 = nn.BatchNorm2d(embed_dim)
        
    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.proj2(x)
        x = self.norm2(x)    
        x = x.permute(0,2,3,1)
        return x




class StaticLocalBranch(nn.Module):
    def __init__(self, dim, r):
        super().__init__()
        self.r = r
        self.GraphConv = GraphConv2d(dim, dim, conv='local', act='GraphLU', norm='batch', bias=True)

    def forward(self, x):
        x = self.GraphConv(x)
        return x.contiguous()




class DynamicGlobalBranch(nn.Module):
    def __init__(self, dim, r):
        super().__init__()
        self.r = r
        self.GraphConv = GraphConv2d(dim, dim, conv='global', act='GraphLU', norm='batch', bias=True)
        self.dilated_knn_graph = DenseDilatedKnnGraph(k=9, dilation=2)

    def forward(self, x):
        B, C, H, W = x.shape
        y = None
        y = F.avg_pool2d(x, self.r, self.r)
        y = y.reshape(B, C, -1, 1).contiguous()
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x, y)
        x = self.GraphConv(x, edge_index, y)
        return x.reshape(B, -1, H, W).contiguous()




class Node_Feature_Update(nn.Module):
    def __init__(self, dim, r, Dydim=32,proj_drop=0.,):
        super().__init__()


        dynamicdim = self.dynamicdim = Dydim
        staticdim = self.staticdim = dim - dynamicdim

        self.StaticGraph = StaticLocalBranch(staticdim, r)
        self.DyGraph = DynamicGlobalBranch(dynamicdim, r)

        self.get_le = nn.Conv2d(dynamicdim + staticdim * 2, dynamicdim + staticdim * 2, kernel_size=3, stride=1, padding=1,bias=False, groups=dynamicdim + staticdim * 2)

        self.proj = nn.Conv2d(dynamicdim+staticdim*2, dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)

    def get_lepe(self, x, func):
        lepe = func(x)
        return lepe
        
    def forward(self, x):

        x = x.permute(0, 3, 1, 2)

        static_branch = x[:,:self.staticdim,:,:].contiguous()
        static_branch = self.StaticGraph(static_branch)

        dynamic_branch = x[:,self.staticdim:,:,:].contiguous()
        dynamic_branch = self.DyGraph(dynamic_branch)

        x = torch.cat((static_branch, dynamic_branch), dim=1)

        lepe = self.get_lepe(x, self.get_le)

        x = x + lepe
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x



class PVG_Block(nn.Module):

    def __init__(self, dim, r ,Dydim=32, mlp_ratio=4.,drop=0.,drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,use_layer_scale=False, layer_scale_init_value=1e-5,):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        
        self.graph = Node_Feature_Update(dim, r,Dydim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.use_layer_scale = use_layer_scale

        if self.use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)


    def forward(self, x):
        k = self.use_layer_scale
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.graph(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.graph(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class PVGraph(nn.Module):
    def __init__(self, img_size=224,in_chans=3, Dydim=None, num_classes=1000, embed_dims=None, depths=None,mlp_ratio=4.,drop_rate=0.,
                 drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,act_layer=None, weight_init='', use_layer_scale=True, layer_scale_init_value=1e-6,**kwargs):
        
        super().__init__()
        st2_idx = sum(depths[:1])
        st3_idx = sum(depths[:2])
        st4_idx = sum(depths[:3])
        depth = sum(depths)
        
        self.num_classes = num_classes
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)


        act_layer = GraphLU


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule


        self.patch_embed = FirstPatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0])
        self.num_patches1 = num_patches = img_size // 4
        self.pos_embed1 = nn.Parameter(torch.zeros(1, num_patches, num_patches, embed_dims[0]))
        self.blocks1 = nn.Sequential(*[
            PVG_Block(dim=embed_dims[0],r=2, Dydim=Dydim[i], mlp_ratio=mlp_ratio,  drop=drop_rate,drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(0, st2_idx)])



        self.patch_embed2 = embed_layer(kernel_size=3, stride=2, padding=1, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.num_patches2 = num_patches = num_patches // 2
        self.pos_embed2 = nn.Parameter(torch.zeros(1, num_patches, num_patches, embed_dims[1]))
        self.blocks2 = nn.Sequential(*[
            PVG_Block(dim=embed_dims[1], r=2, Dydim=Dydim[i], mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i],norm_layer=norm_layer, act_layer=act_layer)
            for i in range(st2_idx,st3_idx)])



        self.patch_embed3 = embed_layer(kernel_size=3, stride=2, padding=1, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.num_patches3 = num_patches = num_patches // 2
        self.pos_embed3 = nn.Parameter(torch.zeros(1, num_patches, num_patches, embed_dims[2]))
        self.blocks3= nn.Sequential(*[
            PVG_Block(dim=embed_dims[2],r=1, Dydim=Dydim[i], mlp_ratio=mlp_ratio,  drop=drop_rate,drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                  use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value,
                  )
            for i in range(st3_idx, st4_idx)])


        self.patch_embed4 = embed_layer(kernel_size=3, stride=2, padding=1, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.num_patches4 = num_patches = num_patches // 2
        self.pos_embed4 = nn.Parameter(torch.zeros(1, num_patches, num_patches, embed_dims[3]))
        self.blocks4 = nn.Sequential(*[
            PVG_Block(dim=embed_dims[3],r=1, Dydim=Dydim[i], mlp_ratio=mlp_ratio,  drop=drop_rate,drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                  use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value,
                  )
            for i in range(st4_idx,depth)])


        
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)
        trunc_normal_(self.pos_embed3, std=.02)
        trunc_normal_(self.pos_embed4, std=.02)
        
        self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
    
    def _get_pos_embed(self, pos_embed, num_patches_def, H, W):
        if H * W == num_patches_def * num_patches_def:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").permute(0, 2, 3, 1)

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, H, W, C = x.shape
        x = x + self._get_pos_embed(self.pos_embed1, self.num_patches1, H, W)
        x = self.blocks1(x)
        
        x = x.permute(0, 3, 1, 2)       
        x = self.patch_embed2(x)
        B, H, W, C = x.shape
        x = x + self._get_pos_embed(self.pos_embed2, self.num_patches2, H, W)
        x = self.blocks2(x)
        
        x = x.permute(0, 3, 1, 2)  
        x = self.patch_embed3(x)
        B, H, W, C = x.shape
        x = x + self._get_pos_embed(self.pos_embed3, self.num_patches3, H, W)
        x = self.blocks3(x)
        
        x = x.permute(0, 3, 1, 2)  
        x = self.patch_embed4(x)
        B, H, W, C = x.shape
        x = x + self._get_pos_embed(self.pos_embed4, self.num_patches4, H, W)
        x = self.blocks4(x)
        x = x.flatten(1,2)

        x = self.norm(x)
        return x.mean(1)
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x




def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0.):
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
    elif isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


@register_model
def pvg_s(pretrained=False, **kwargs):
    depths = [2, 3, 12, 4]
    embed_dims = [96, 192, 320, 384]

    model = PVGraph(img_size=224,
                    depths=depths,
                    embed_dims=embed_dims,
                    Dydim=Dydim_S,
                    use_layer_scale=True, layer_scale_init_value=1e-6,
                    **kwargs)
    return model


@register_model
def pvg_m(pretrained=False, **kwargs):
    depths = [4, 6, 14, 6]
    embed_dims = [96, 192, 384, 512]

    model = PVGraph(img_size=224,
                    depths=depths,
                    embed_dims=embed_dims,
                    Dydim=Dydim_M,
                    use_layer_scale=True, layer_scale_init_value=1e-6,
                    **kwargs)
    return model


@register_model
def pvg_l(pretrained=False, **kwargs):
    depths = [4, 6, 24, 8]
    embed_dims = [96, 192, 448, 640]

    model = PVGraph(img_size=224, depths=depths,
                    embed_dims=embed_dims,
                    Dydim=Dydim_L,
                    use_layer_scale=True, layer_scale_init_value=1e-6,
                    **kwargs)
    return model



