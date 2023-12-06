# original vision transformer from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from torch.nn.utils import clip_grad_norm_
from scipy.optimize import brentq
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


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


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n,_, h = *x.shape, self.heads
        #print(x.shape)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        #print(qkv[0].shape)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class PositionalEncoding(nn.Module):
    # https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x = x + self.pe[:x.size(0), :]
        # x.size(0): batch size, x.size(1): length of sequence
        #print(self.pe[:, :x.size(1)].shape,x.shape)
        #print(self.pe.shape,self.pe[:, :x.size(1)].shape,x.shape)
        x = x + self.pe[:, :x.size(1)]
        return x


class ViT(nn.Module):
    """
    input_size: number of inputs
    input_dim: number of channels in input
    dim: Last dimension of output tensor after linear transformation nn.Linear(..., dim).
    depth: Number of Transformer blocks.
    heads: Number of heads in Multi-head Attention layer.
    mlp_dim: Dimension of the MLP (FeedForward) layer.
    dropout: Dropout rate.
    emb_dropout: Embedding dropout rate.
    pool: either cls token pooling or mean pooling
    """

    def __init__(self, *, input_dim=362, output_dim=3, dim=1024, depth=6, heads=16, mlp_dim=2048, pool='cls', dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()

        self.project = nn.Linear(input_dim, 3)
        self.project2 = nn.Linear(181,dim)
        self.pos_encoder = PositionalEncoding(dim)
        # self.pos_embedding = nn.Parameter(torch.randn(1, input_size + 1, dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 64),
            nn.LayerNorm(64),
            nn.Linear(64, output_dim)
        )
        self.loss = self.GE2E_softmax_loss
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        #x= x.unsqueeze(1)
        
        #x = self.project(x)
        x = torch.reshape(x,(x.shape[0],6,181))
        #print(x.shape)
        x = self.project2(x)
        b, n,_ = x.shape

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]

        x = self.pos_encoder(x)
        # x += self.pos_embedding[:, :(n)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        # return self.mlp_head(x)
        x = self.tanh(self.mlp_head(x))
        x = x / torch.norm(x, dim=1, keepdim=True)

        return {"style":x}

    def similarity_matrix(self, embeds):
        """
        Computes the similarity matrix according the section 2.1 of GE2E.

        :param embeds: the embeddings as a tensor of shape (players_per_batch,
        games_per_player, embedding_size)
        :return: the similarity matrix as a tensor of shape (players_per_batch,
        games_per_player, players_per_batch)
        """
        players_per_batch, games_per_player = embeds.shape[:2]

        # Inclusive centroids (1 per speaker). Cloning is needed for reverse differentiation
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / torch.norm(centroids_incl, dim=2, keepdim=True)

        # Exclusive centroids (1 per utterance)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (games_per_player - 1)
        centroids_excl = centroids_excl.clone() / torch.norm(centroids_excl, dim=2, keepdim=True)

        # Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
        # product of these vectors (which is just an element-wise multiplication reduced by a sum).
        # We vectorize the computation for efficiency.
        sim_matrix = torch.zeros(players_per_batch, games_per_player,
                                 players_per_batch).to(self.loss_device)
        mask_matrix = 1 - np.eye(players_per_batch, dtype=np.int)
        for j in range(players_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)

        # Even more vectorized version (slower maybe because of transpose)
        # sim_matrix2 = torch.zeros(speakers_per_batch, speakers_per_batch, utterances_per_speaker
        #                           ).to(self.loss_device)
        # eye = np.eye(speakers_per_batch, dtype=np.int)
        # mask = np.where(1 - eye)
        # sim_matrix2[mask] = (embeds[mask[0]] * centroids_incl[mask[1]]).sum(dim=2)
        # mask = np.where(eye)
        # sim_matrix2[mask] = (embeds * centroids_excl).sum(dim=2)
        # sim_matrix2 = sim_matrix2.transpose(1, 2)

        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix
    def GE2E_softmax_loss(self, sim_matrix, players_per_batch, games_per_player):

        # colored entries in paper
        sim_matrix_correct = torch.cat([sim_matrix[i * games_per_player:(i + 1) * games_per_player, i:(i + 1)] for i in range(players_per_batch)])
        # softmax loss
        loss = -torch.sum(sim_matrix_correct - torch.log(torch.sum(torch.exp(sim_matrix), axis=1, keepdim=True) + 1e-6)) / (players_per_batch * games_per_player)
        return loss

    def loss(self, embeds,ground_truth):
        """
        Computes the softmax loss according the section 2.1 of GE2E.

        :param embeds: the embeddings as a tensor of shape (players_per_batch,
        games_per_player, embedding_size)
        :return: the loss and the EER for this batch of embeddings.
        """
        players_per_batch, games_per_player = embeds.shape[:2]

        # Loss
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((players_per_batch * games_per_player,
                                         players_per_batch))
        
        # target = torch.from_numpy(ground_truth).long().to(self.loss_device)
        # loss = self.loss_fn(sim_matrix, target)
        loss = self.loss(sim_matrix, players_per_batch, games_per_player)

        # EER (not backpropagated)
        with torch.no_grad():
            
            labels = ground_truth
            preds = sim_matrix.detach().cpu().numpy()

            # Snippet from https://yangcha.github.io/EER-ROC/
            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        return loss, eer