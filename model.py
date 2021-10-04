from torch import nn 

class DeepTCDR (nn.Module):
    
    def __init__(self, batch_size: int):
        super(DeepTCDR, self).__init__()
        
        # Transformer block 
        TB1 = nn.Transformer(d_model=batch_size,
                             nhead=8,
                             num_encoder_layers=6,
                             num_decoder_layers=6,
                             dim_feedforward=2048,
                             dropout=0.1,
                             activation="relu")
        
    def forward(self, x):
        
        # Transformer block 
        x = self.TB1(x)
        
        return x
    
         
        
        
