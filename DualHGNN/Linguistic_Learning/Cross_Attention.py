import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionModule(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(CrossAttentionModule, self).__init__()
        # 初始化线性层，将输入映射到Q, K, V空间
        self.query_linear = nn.Linear(hidden_dim, hidden_dim)
        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, hidden_dim)

        # Cross Attention部分
        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

        # LayerNorm
        self.norm = nn.LayerNorm(hidden_dim)


    def forward(self, syntax_hidden, struct_hidden):
        # Cross Attention
        Q = self.query_linear(syntax_hidden)  # Query
        K = self.key_linear(struct_hidden)  # Key
        V = self.value_linear(struct_hidden)  # Value

        # 计算Cross Attention
        attn_output, _ = self.multihead_attention(Q, K, V)

        # 残差连接 + Add & LayerNorm
        output = self.norm(attn_output + Q)

        return output


# # 假设有两个张量
# syntax_hidden = torch.randn(5, 512)
# struct_hidden = torch.randn(5, 512)

# # 创建CrossAttention模块
# hidden_dim = 512
# num_heads = 8
# cross_attention_module = CrossAttentionModule(hidden_dim, num_heads)
#
# # 前向传播
# output1 = cross_attention_module(syntax_hidden, struct_hidden)
# output2 = cross_attention_module(struct_hidden, syntax_hidden)
#
# # 沿最后一个维度拼接（dim=2 表示最后一个维度）
# feature_output = torch.cat((output1, output2), dim=1)
# print(feature_output.shape)