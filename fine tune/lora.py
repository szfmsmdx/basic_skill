import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LinearLoRALayer(nn.Module):
    def __init__(self, 
        in_features, 
        out_features,
        merge=False,
        rank=8,
        lora_alpha=16,
        dropout=0.1,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.merge = merge
        self.rank = rank

        self.linear = nn.Linear(in_features, out_features)
        
        if rank > 0:
            # 这里是为了标记 lora_a 和 lora_b 是可训练的参数
            self.lora_a = nn.Parameter(
                torch.zeros(out_features, rank)
            )
            # lora_a 需要初始化为 高斯分布
            nn.init.kaiming_normal_(self.lora_a, a=0.01)

            self.lora_b = nn.Parameter(
                torch.zeros(rank, in_features)
            )
            self.scale = lora_alpha / rank

            # linear 需要设置为不可以训练
            self.linear.weight.requires_grad = False
            self.linear.bias.requires_grad = False
        
        self.dropout = nn.Dropout(
            dropout
        ) if dropout > 0 else nn.Identity()

        # 如果采用 merge 进行推理，
        # 那么会把 lora_a 和 lora_b 两个小矩阵的参数直接放到 linear.weight 中
        if merge:
            self.merge_weight()

    
    def forward(self, X):
        # X shape is (batch, seq_len, in_feature)
        # lora_a 是 out_features * rank
        if self.rank > 0 and self.merge:
            output = self.linear(X) + self.scale * ( X @ (self.lora_a @ self.lora_b).T )
        elif self.rank > 0 and not self.merge:
            output = self.linear(X)
        else:
            output = self.linear(X)
        
        return self.dropout(output)

    def merge_weight(self, ):
        if self.merge and self.rank > 0:
            self.linear.weight.data += self.scale * (self.lora_a @ self.lora_b)
    
    def unmerge_weight(self, ):
        if self.rank > 0:
            self.linear.weight.data -= self.scale * (self.lora_a @ self.lora_b)


# 写一段测试代码
# Test the LoRALinear layer
batch_size = 32
seq_len = 128
in_features = 768
out_features = 512
rank = 8
lora_alpha = 16
dropout = 0.1

# Create a test input
x = torch.randn(batch_size, seq_len, in_features)

# Test regular mode (no merge)
lora_layer = LinearLoRALayer(
    in_features=in_features,
    out_features=out_features,
    rank=rank,
    lora_alpha=lora_alpha,
    dropout=dropout,
    merge=False
)

# Forward pass
output = lora_layer(x)
print(f"Output shape (no merge): {output.shape}")  # Should be [batch_size, seq_len, out_features]

# Test merged mode
lora_layer_merged = LinearLoRALayer(
    in_features=in_features,
    out_features=out_features,
    rank=rank,
    lora_alpha=lora_alpha,
    dropout=dropout,
    merge=True
)

# Forward pass with merged weights
output_merged = lora_layer_merged(x)
print(f"Output shape (merged): {output_merged.shape}")  # Should be [batch_size, seq_len, out_features]

# Test weight merging/unmerging
lora_layer.merge_weight()
output_after_merge = lora_layer(x)
lora_layer.unmerge_weight()
output_after_unmerge = lora_layer(x)

print("Max difference after merge/unmerge cycle:", 
      torch.max(torch.abs(output - output_after_unmerge)).item())