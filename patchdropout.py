import torch

class PatchDropout(torch.nn.Module):
    """ 
    Implements PatchDropout: https://arxiv.org/abs/2208.07220
    """
    def __init__(self, keep_rate=0.5, sampling="uniform", token_shuffling=False):
        super().__init__()
        assert 0 < keep_rate <=1, "The keep_rate must be in (0,1]"
        
        self.keep_rate = keep_rate
        self.sampling = sampling
        self.token_shuffling = token_shuffling

    def forward(self, x, force_drop=False):
        """
        If force drop is true it will drop the tokens also during inference.
        """
        if not self.training and not force_drop: return x        
        if self.keep_rate == 1: return x

        # batch, length, dim
        N, L, D = x.shape
        
        # making cls mask (assumes that CLS is always the 1st element)
        cls_mask = torch.zeros(N, 1, dtype=torch.int64, device=x.device)
        # generating patch mask
        patch_mask = self.get_mask(x)

        # cat cls and patch mask
        patch_mask = torch.hstack([cls_mask, patch_mask])
        
        # gather tokens
        x = torch.gather(x, dim=1, index=patch_mask.unsqueeze(-1).repeat(1, 1, D))

        return x
    
    def get_mask(self, x):
        if self.sampling == "uniform":
            return self.uniform_mask(x)
        else:
            return NotImplementedError(f"PatchDropout does not support {self.sampling} sampling")
    
    def uniform_mask(self, x):
        """
        Returns an iid-mask using uniform sampling
        """
        N, L, D = x.shape
        _L = L -1 # patch length (without CLS)
        
        keep = int(_L * self.keep_rate)
        patch_mask = torch.rand(N, _L, device=x.device)
        patch_mask = torch.argsort(patch_mask, dim=1) + 1
        patch_mask = patch_mask[:, :keep]
        if not self.token_shuffling:
            patch_mask = patch_mask.sort(1)[0]
        return patch_mask