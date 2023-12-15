import torch
import torchvision

import tokens

class SensedPatchDropout(torch.nn.Module):
    def __init__(self,
                 tokens,
                 ratio,
                 train_sampling='oracle', 
                 inference_sampling='oracle',
                 basis='Identity',
                 sensors='128',
                 sensing_patch_size=4, 
                 token_shuffling=False):
        super().__init__()
        
        self.tokens = tokens
        self.ratio = ratio
        self.train_sampling = train_sampling
        self.inference_sampling = inference_sampling
        self.basis = basis
        self.sensors = sensors
        self.sensing_patch_size = sensing_patch_size
        self.strategy = 'ranking'
        self.token_shuffling = token_shuffling

    def forward(self, x):
        if (self.train_sampling if self.training else self.inference_sampling) == 'oracle': return x        

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
    
    def update_sensing_mask(self, x):
        sampling = self.train_sampling if self.training else self.inference_sampling
        if sampling in ['r']:
            per_image = True
            downscaled_x = torchvision.transforms.functional.resize(x, size=(32, 32), antialias=False)
            if per_image:
                model = tokens.get_model(fit_type=sampling, basis=self.basis, modes=1, sensors=self.sensors)
                self.token_mask = tokens.fit_mask_per_image(model=model, x=downscaled_x, sensing_patch_size=self.sensing_patch_size, 
                                                            tokens=int(self.ratio * self.tokens), strategy=self.strategy)
            else:
                n = x.size(0)
                model = tokens.get_model(fit_type=sampling, basis=self.basis, modes=n, 
                                         sensors=self.sensors)
                self.token_mask = tokens.fit_mask(model=model, fit_type=sampling, x=downscaled_x, sensing_patch_size=self.sensing_patch_size,
                                                  tokens=int(self.ratio * self.tokens), strategy=self.strategy).expand((n, -1))
        return

    def get_mask(self, x):
        sampling = self.train_sampling if self.training else self.inference_sampling
        if sampling == 'random':
            return self.random_mask(x)
        elif sampling in ['r']:
            return self.sensed_mask(x)
        else:
            return NotImplementedError(f"SensedPatchDropout does not support {sampling} sampling")
    
    def random_mask(self, x):
        """
        Returns an iid-mask using uniform sampling
        """
        n, l, d = x.shape
        _l = l - 1 # patch length (without CLS)
        
        patch_mask = torch.rand(n, _l, device=x.device)
        patch_mask = torch.argsort(patch_mask, dim=1) + 1
        patch_mask = patch_mask[:, :self.tokens]
        if not self.token_shuffling:
            patch_mask = patch_mask.sort(1)[0]
        return patch_mask
    
    def sensed_mask(self, x):
        n, l, d = x.shape
        _l = l - 1 # patch length (without CLS)

        random_tokens = int((1 - self.ratio) * self.tokens)
        sensed_tokens = int(self.ratio * self.tokens)

        sensed_mask = self.token_mask.argwhere()[:, 1].reshape(n, sensed_tokens)

        zeros = (self.token_mask == 0).argwhere()[:, 1].reshape(n, _l - sensed_tokens)
        shuffled_zeros = torch.rand(n, _l - sensed_tokens)
        sorted_zeros = torch.argsort(shuffled_zeros, dim=1)
        random_mask = torch.gather(zeros, 1, sorted_zeros)[:, :random_tokens]
        patch_mask = (torch.cat((random_mask, sensed_mask), dim=1) + 1).to(x.device)
        if not self.token_shuffling:
            patch_mask = patch_mask.sort(1)[0]
        return patch_mask
        