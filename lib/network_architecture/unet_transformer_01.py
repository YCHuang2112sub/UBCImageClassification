import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class Unet(nn.Module):
    def __init__(self, in_channels, classes):
        super(Unet, self).__init__()
        self.n_channels = in_channels
        self.n_classes =  classes

        self.inc = InConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x, (x1, x2, x3, x4, x5)
    
class BridgingModel(nn.Module):
    def __init__(self):
        super(BridgingModel, self).__init__()

        # self.mlp = nn.Sequential(
        #     nn.Linear(hidden_d, mlp_ratio * hidden_d),
        #     nn.GELU(),
        #     nn.Linear(mlp_ratio * hidden_d, hidden_d)
        # )

        self.trans1_1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1)
        )
        self.trans1_2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1)
        )
        self.trans1_3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1)
        )
        self.trans1_4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=2, padding=1)
        )

        self.trans2_2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1)
        )
        self.trans2_3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1)
        )
        self.trans2_4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=2, padding=1)
        )

    def forward(self, x1, x2, x3, x4, x5):
        # print(a1(x1).shape)
        t1_1 = torch.cat( (x2[:, :64, :, :] + self.trans1_1(x1), x2[:, 64:, :, :] ), dim=1)
        t1_2 = torch.cat( (x3[:, :128, :, :] + self.trans1_2(t1_1), x3[:, 128:, :, :] ), dim=1)
        t1_3 = torch.cat( (x4[:, :256, :, :] + self.trans1_3(t1_2), x4[:, 256:, :, :] ), dim=1)
        t1_4 = torch.cat( (x5[:, :512, :, :] + self.trans1_4(t1_3), x5[:, 512:, :, :] ), dim=1)

        t2_2 = torch.cat( (t1_2[:, :128, :, :] + self.trans2_2(x2), t1_2[:, 128:, :, :] ), dim=1)
        t2_3 = torch.cat( (t1_3[:, :256, :, :] + self.trans2_3(t2_2), t1_3[:, 256:, :, :] ), dim=1)
        t2_4 = torch.cat( (t1_4[:, :512, :, :] + self.trans2_4(t2_3), t1_4[:, 512:, :, :] ), dim=1)

        f1, f2 = t2_4, t1_4

        return f1, f2
    
class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        # self.norm1 = nn.LayerNorm(hidden_d)
        # self.mhsa = MyMSA(hidden_d, n_heads)
        
        self.norm1_q = nn.LayerNorm(hidden_d)
        self.norm1_k = nn.LayerNorm(hidden_d)
        self.norm1_v = nn.LayerNorm(hidden_d)
        self.multihead_attn = nn.MultiheadAttention(hidden_d, n_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, q, k, v):
        # x = v
        # out = x + self.mhsa(self.norm1(x))
        q1, k1, v1 = self.norm1_q(q), self.norm1_k(k), self.norm1_v(v)
        attn_output, attn_output_weights = self.multihead_attn(q1, k1, v1, average_attn_weights=False)
        out = v + attn_output
        
        out = out + self.mlp(self.norm2(out))
        return out

class FeatureTransformer(nn.Module):
    def __init__(self, data_dim=(256,512), hidden_d=512, n_heads=16, out_d=5) -> None:
        super(FeatureTransformer, self).__init__()
        
        assert(len(data_dim) == 2)

        self.data_dim = data_dim # (Length, embedding_dim)
        self.seq_len, self.emb_dim = data_dim
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.out_d = out_d

        # 1) Input transformation
        self.linear_mapper = nn.Linear(data_dim[1], hidden_d)

        # 2) Positional embedding
        self.positional_embeddings = nn.Parameter(torch.randn(1, self.seq_len, hidden_d))
        # self.pos_embeddings = self.get_positional_embeddings(self.seq_len, hidden_d)


        # 3) Transformer encoder blocks
        # self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        self.block0 = MyViTBlock(hidden_d, n_heads)
        self.block1 = MyViTBlock(hidden_d, n_heads)
        self.block2 = MyViTBlock(hidden_d, n_heads)
        self.block3 = MyViTBlock(hidden_d, n_heads)

        # 4) Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, hidden_d),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(self.hidden_d, out_d),
            # nn.Softmax(dim=-1)
        )


    def forward(self, x5, v1, v2):
        B_size = x5.shape[0]

        tokens = self.linear_mapper(x5)
        
        # Adding classification token to the tokens
        # tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])
        
        # Adding positional embedding
        pos_embed = self.positional_embeddings.repeat(B_size, 1, 1)
        out = tokens + pos_embed  
        
        # Transformer Blocks
        # for block in self.blocks:
        #     out2 = block(out, out, out)
            
        #     out = out2
        x_stage1 = self.block0(out, v1, v1) # Q, K, V
        x_stage2 = self.block1(x_stage1, x_stage1, x_stage1)
        x_stage3 = self.block2(x_stage2, v2, v2)
        x_stage4 = self.block3(x_stage3, x_stage3, x_stage3)

        out = x_stage4
            
        # Getting the classification token only
        out = out[:, 0]
        
        return self.mlp(out), out # Map to output dimension, output category distribution
            
    
    def get_positional_embeddings(self, sequence_length, d):
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return result
        

def get_unet_transformer_model_output(X, unet_model, bridging_model, feature_transformer_model):
    unet_out, hidden_features = unet_model(X)
    (x1, x2, x3, x4, x5) = hidden_features
    fs = bridging_model(x1, x2, x3, x4, x5)
    B, C, H, W = fs[0].shape
    vs = [x.permute(0,2,3,1).reshape(B, H*W, C) for x in fs]

    x5 = hidden_features[4].permute(0,2,3,1).reshape(B, H*W, C)
    y_pred, out = feature_transformer_model(x5, vs[0], vs[1])
    return y_pred, out, unet_out

