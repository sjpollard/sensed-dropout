import torch

import tokens

class SensorDropout(torch.nn.Module):
    def __init__(self,
                 tokens,
                 train_sampling='r', 
                 inference_sampling='oracle', 
                 basis='Identity',
                 modes='128',
                 sensors='128',
                 l1_penalty=0.001,
                 patch=4,
                 strategy='ranking',
                 token_shuffling=False):
        super().__init__()
        
        self.tokens = tokens
        self.train_sampling = train_sampling
        self.inference_sampling = inference_sampling
        self.basis = basis
        self.modes = modes
        self.sensors = sensors
        self.l1_penalty = l1_penalty
        self.patch = patch
        self.strategy = strategy
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
    
    def update_mask(self, x, y):
        sampling = None
        if self.training and self.train_sampling in ['r', 'c']:
            sampling = self.train_sampling
        elif not self.training and self.inference_sampling in ['r']:
            sampling = self.inference_sampling
        if sampling != None:
            model = tokens.get_model(fit_type=sampling, basis=self.basis, modes=self.modes, 
                                     sensors=self.sensors, l1_penalty=self.l1_penalty)
            print(tokens.fit_mask(model, sampling, x, y, self.patch, self.tokens, self.strategy))
        return

    def get_mask(self, x):
        if self.sampling == 'uniform':
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