import torch
from torch import nn
from einops import rearrange, repeat
from .rn import Scale_Embedding
import math

MIN_NUM_PATCHES = 16


class MultiHeadDotProductAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = (dim / heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, keep_rate, mask=None):
        b, n, c, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        # if mask is not None:
        #     mask = F.pad(mask.flatten(1), (1, 0), value=True)
        #     assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
        #     mask = mask[:, None, :] * mask[:, :, None]
        #     dots.masked_fill_(~mask, float('-inf'))
        #     del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        # EVIT
        left_tokens = n-1
        if keep_rate < 1:
            left_tokens = math.ceil(keep_rate * left_tokens)
            cls_attn = attn[:, :, 0, 1:]  # [B, H, N-1]
            cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
            _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)  # [B, left_tokens]
            idx, _ = torch.sort(idx)
            index = idx.unsqueeze(-1).expand(-1, -1, c)  # [B, left_tokens, C]

            return x, index, idx, cls_attn, left_tokens

        return out, None, None, None, left_tokens


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Encoder1DBlock(nn.Module):
    def __init__(self, input_shape, heads, mlp_dim, dtype=torch.float32, dropout_rate=0.1, attention_dropout_rate=0.1,
                 deterministic=True):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.dtype = dtype
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.deterministic = deterministic
        self.input_shape = input_shape
        self.layer_norm_input = nn.LayerNorm(input_shape)
        self.layer_norm_out = nn.LayerNorm(input_shape)

        self.attention = MultiHeadDotProductAttention(input_shape, heads=heads)
        self.mlp = FeedForward(input_shape, mlp_dim, dropout_rate)
        self.drop_out_attention = nn.Dropout(attention_dropout_rate)

    def forward(self, inputs, keep_rate):
        x = self.layer_norm_input(inputs)
        x, index, idx, cls_attn, left_tokens = self.attention(x, keep_rate)
        x = self.drop_out_attention(x)
        x = x + inputs

        if index is not None:
            # B, N, C = x.shape
            non_cls = x[:, 1:]
            x_others = torch.gather(non_cls, dim=1, index=index)    # [B, left_tokens, C]
            x = torch.cat([x[:, 0:1], x_others], dim=1)     # [B, N+1, C] ->  [B, left_tokens+1, C]

        y = self.layer_norm_out(x)
        y = self.mlp(y)
        return x + y, left_tokens, idx


class Encoder(nn.Module):
    def __init__(self, input_shape, num_layers, heads, mlp_dim, inputs_positions=None, dropout_rate=0.1, train=False):
        super().__init__()
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.inputs_positions = inputs_positions
        self.dropout_rate = dropout_rate
        self.train_flag = train
        self.encoder_norm = nn.LayerNorm(input_shape)
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([Encoder1DBlock(input_shape, heads, mlp_dim)]))

        self.keep_rate = (1, ) * 12
        # self.keep_rate = (1, 1, 1, 0.9) + (1, 1, 0.9) + (1, 1, 0.9) + (1, 1)    # 196 -> 177 -> 160 -> 144

    def forward(self, img, mask=None):
        x = img
        left_tokens = []
        idxs = []

        for i, layer in enumerate(self.layers):
            x, left_token, idx = layer[0](x, self.keep_rate[i])
            left_tokens.append(left_token)
            idxs.append(idx)

        return self.encoder_norm(x), left_tokens, idxs


class ViTPatch(nn.Module):
    def __init__(self, *, image_size, patch_size, hidden_size, num_classes, depth, heads,
                 mlp_dim, channels=3, dropout=0., emb_dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective. try decreasing your patch size'

        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.embedding = nn.Conv2d(channels, hidden_size, patch_size, patch_size)
        self.scale = Scale_Embedding()

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_size))
        self.cls = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Encoder(hidden_size, depth, heads, mlp_dim, dropout_rate=dropout)
        self.to_cls_token = nn.Identity()
        # self.mlp_head = nn.Linear(hidden_size, num_classes)

    def forward(self, img, mask=None):
        x1 = self.embedding(img)
        x2 = self.scale(img)
        x = (x1 + x2) / 2

        x = rearrange(x, 'b c h w  -> b (h w) c')
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x, left_tokens, idxs = self.transformer(x)

        # x1 = self.to_cls_token(x[:, 0])
        # return x1, x[:, 1:]

        return x, left_tokens, idxs


class Self_Attention(nn.Module):
    def __init__(self, d_model=768, cls_number=100, pretrained=True):
        super(Self_Attention, self).__init__()
        self.model = ViTPatch(
            image_size=224,
            patch_size=16,
            hidden_size=d_model,
            num_classes=cls_number,
            depth=12,
            heads=12,
            mlp_dim=3072,
            dropout=0.1,
            emb_dropout=0.1
        )
        if pretrained:
            checkpoint = torch.load("./sam_ViT-B_16.pth")
            cur = self.model.state_dict()
            new = {k: v for k, v in checkpoint.items() if k in cur.keys() and 'mlp_head' not in k}
            cur.update(new)
            self.model.load_state_dict(cur)

    def forward(self, x):
        sa_fea, left_tokens, idxs = self.model(x)
        return sa_fea, left_tokens, idxs

