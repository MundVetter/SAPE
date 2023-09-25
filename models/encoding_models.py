import wandb
from custom_types import *
import abc

from custom_types import OptimizerW, nn, nnf, torch
from utils import train_utils
import copy
import math

#note that 1 is not a prime number
PRIMES = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]


class ModelParams:
    def fill_args(self, **kwargs):
        for key, item in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, item)

    def __init__(self, **kwargs):
        self.domain_dim = 2
        self.num_frequencies = 256
        self.std = 20
        self.num_layers = 3
        self.hidden_dim = 256
        self.output_channels = 3
        self.use_id_encoding = False
        self.bn = False
        self.fill_args(**kwargs)


class MLP(nn.Module):

    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, layers: Union[List[int], Tuple[int, ...]], bn = False):
        super(MLP, self).__init__()
        layers_ = []
        for i in range(len(layers) - 1):
            layers_.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                layers_.append(nn.ReLU(True))
                if bn:
                    layers_.append(nn.BatchNorm1d(layers[i + 1], momentum = 1))
        self.model = nn.Sequential(*layers_)

class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()
    def forward(self, x):
        return torch.cos(x)


class EncodingLayer(nn.Module, abc.ABC):

    @property
    @abc.abstractmethod
    def output_channels(self) -> int:
        raise NotImplemented


class EncodedMlpModel(nn.Module, abc.ABC):

    @property
    def domain_dim(self):
        return self.opt.domain_dim

    @property
    def encoding_dim(self):
        if self.opt.use_id_encoding:
            return self.encode.output_channels + self.opt.domain_dim
        else:
            return self.encode.output_channels

    @abc.abstractmethod
    def get_encoding_layer(self) -> EncodingLayer:
        raise NotImplemented

    def get_mlp_model(self) -> nn.Module:
        dims = [self.encoding_dim] + self.opt.num_layers * [self.opt.hidden_dim] + [self.opt.output_channels]
        return MLP(dims, bn = self.opt.bn)

    def mlp_forward(self, x: T):
        return self.model(x)

    def get_encoding(self, x: T) -> T:
        encoding = self.encode(x)
        if self.opt.use_id_encoding:
            return torch.cat((x, encoding), dim=-1)
        else:
            return encoding

    def forward(self, x: T, *args, **kwargs) -> T:
        base_code = self.get_encoding(x)
        if 'override_mask' in kwargs:
            base_code = base_code * kwargs['override_mask']
        out = self.mlp_forward(base_code)
        return out

    def __init__(self, opt: ModelParams):
        super(EncodedMlpModel, self).__init__()
        self.opt = opt
        self.encode = self.get_encoding_layer()
        self.model = self.get_mlp_model()


class IdEncoding(EncodingLayer):

    def forward(self, x):
        return x

    @property
    def output_channels(self) -> int:
        return self.domain_dim

    def __init__(self, domain_dim: int):
        super(IdEncoding, self).__init__()
        self.domain_dim = domain_dim


class BaseModel(EncodedMlpModel):

    def get_encoding_layer(self) -> EncodingLayer:
        return IdEncoding(self.opt.domain_dim)


class FourierFeatures(EncodingLayer, abc.ABC):

    @property
    def output_channels(self) -> int:
        return self.num_frequencies * 2

    def forward(self, x: T):
        shape = x.shape[:-1]
        if x.dim() != 2:
            x = x.view(-1, self.domain_dim)
        x = x * 2 * np.pi
        out = torch.matmul(x, self.frequencies)
        out = torch.sin(out), torch.cos(out)
        out = torch.stack(out, dim=2).view(*shape, self.output_channels)
        return out

    @abc.abstractmethod
    def init_frequencies(self, std: float) -> T:
        raise NotImplemented

    def __init__(self, domain_dim: int, num_frequencies: int, std: float):
        super(FourierFeatures, self).__init__()
        self.domain_dim = domain_dim
        self.num_frequencies: int = num_frequencies
        frequencies = self.init_frequencies(std)
        self.register_buffer("frequencies", frequencies)


class GaussianRandomFourierFeatures(FourierFeatures):

    def init_frequencies(self, std: float) -> T:
        magnitude = torch.randn(self.num_frequencies) * std
        order = magnitude.abs().argsort(0)
        magnitude = magnitude[order]
        frequencies: T = torch.randn(self.domain_dim, self.num_frequencies)
        frequencies = nnf.normalize(frequencies, p=2, dim=0) * magnitude[None, :]
        return frequencies
    
class Hashgrid(EncodingLayer, abc.ABC):
    def __init__(self, domain_dim: int, L: int, F: int, map_size: int, base_res: int, finest_res: int) -> None:
        super().__init__()
        self.encoder = MultiResHashGrid(dim=domain_dim, n_levels=L, n_features_per_level=F, base_resolution=base_res, log2_hashmap_size=map_size, finest_resolution=finest_res)

        b = math.exp((math.log(finest_res) - math.log(base_res))/(L - 1))
        self.frequencies = torch.floor(base_res * (b ** torch.arange(L))).repeat_interleave(F)

    @property
    def output_channels(self) -> int:
        return self.encoder.output_dim
    
    def forward(self, x: T) -> T:
        return self.encoder(x)

### Adapted from https://github.com/Ending2015a/hash-grid-encoding/blob/master/encoding.py
@torch.no_grad()
def fast_hash(ind: torch.Tensor, primes: torch.Tensor, hashmap_size: int):
  """Hashing function from:
  https://github.com/NVlabs/tiny-cuda-nn/blob/master/include/tiny-cuda-nn/encodings/grid.h#L76-L92
  """
  d = ind.shape[-1]
  ind = (ind * primes[:d]) & 0xffffffff  # uint32
  for i in range(1, d):
    ind[..., 0] ^= ind[..., i]
  return ind[..., 0] % hashmap_size

class _HashGrid(nn.Module):
  def __init__(
    self,
    dim: int,
    n_features: int,
    hashmap_size: int,
    resolution: float
  ):
    super().__init__()
    self.dim = dim
    self.n_features = n_features
    self.hashmap_size = hashmap_size
    self.resolution = resolution

    # you can add more primes for supporting more dimensions
    assert self.dim <= len(PRIMES), \
      f"HashGrid only supports < {len(PRIMES)}-D inputs"

    # create look-up table
    self.embedding = nn.Embedding(hashmap_size, n_features)
    nn.init.uniform_(self.embedding.weight, a=-0.0001, b=0.0001)

    primes = torch.tensor(PRIMES, dtype=torch.int64)
    self.register_buffer('primes', primes, persistent=False)

    # create interpolation binary mask
    n_neigs = 1 << self.dim
    neigs = np.arange(n_neigs, dtype=np.int64).reshape((-1, 1))
    dims = np.arange(self.dim, dtype=np.int64).reshape((1, -1))
    bin_mask = torch.tensor(neigs & (1 << dims) == 0, dtype=bool) # (neig, dim)
    self.register_buffer('bin_mask', bin_mask, persistent=False)

  def forward(self, x: torch.Tensor):
    # x: (b..., dim), torch.float32, range: [0, 1]
    bdims = len(x.shape[:-1])
    x = x * self.resolution
    xi = x.long()
    xf = x - xi.float().detach()
    xi = xi.unsqueeze(dim=-2) # (b..., 1, dim)
    xf = xf.unsqueeze(dim=-2) # (b..., 1, dim)
    # to match the input batch shape
    bin_mask = self.bin_mask.reshape((1,)*bdims + self.bin_mask.shape) # (1..., neig, dim)
    # get neighbors' indices and weights on each dim
    inds = torch.where(bin_mask, xi, xi+1) # (b..., neig, dim)
    ws = torch.where(bin_mask, 1-xf, xf) # (b...., neig, dim)
    # aggregate nehgibors' interp weights
    w = ws.prod(dim=-1, keepdim=True) # (b..., neig, 1)
    # hash neighbors' id and look up table
    hash_ids = fast_hash(inds, self.primes, self.hashmap_size) # (b..., neig)
    neig_data = self.embedding(hash_ids) # (b..., neig, feat)
    return torch.sum(neig_data * w, dim=-2) # (b..., feat)

class MultiResHashGrid(nn.Module):
  def __init__(
    self,
    dim: int,
    n_levels: int = 16,
    n_features_per_level: int = 2,
    log2_hashmap_size: int = 15,
    base_resolution: int = 16,
    finest_resolution: int = 512,
  ):
    """NVidia's hash grid encoding
    https://nvlabs.github.io/instant-ngp/

    The output dimensions is `n_levels` * `n_features_per_level`,
    or your can simply access `model.output_dim` to get the output dimensions

    Args:
      dim (int): input dimensions, supports at most 7D data.
      n_levels (int, optional): number of grid levels. Defaults to 16.
      n_features_per_level (int, optional): number of features per grid level.
        Defaults to 2.
      log2_hashmap_size (int, optional): maximum size of the hashmap of each
        level in log2 scale. According to the paper, this value can be set to
        14 ~ 24 depending on your problem size. Defaults to 15.
      base_resolution (int, optional): coarsest grid resolution. Defaults to 16.
      finest_resolution (int, optional): finest grid resolution. According to
        the paper, this value can be set to 512 ~ 524288. Defaults to 512.
    """
    super().__init__()
    self.dim = dim
    self.n_levels = n_levels
    self.n_features_per_level = n_features_per_level
    self.log2_hashmap_size = log2_hashmap_size
    self.base_resolution = base_resolution
    self.finest_resolution = finest_resolution

    # from paper eq (3)
    b = math.exp((math.log(finest_resolution) - math.log(base_resolution))/(n_levels-1))

    levels = []
    for level_idx in range(n_levels):
      resolution = math.floor(base_resolution * (b ** level_idx))
      hashmap_size = min(resolution ** dim, 2 ** log2_hashmap_size)
      levels.append(_HashGrid(
        dim = dim,
        n_features = n_features_per_level,
        hashmap_size = hashmap_size,
        resolution = resolution
      ))
    self.levels = nn.ModuleList(levels)

    self.input_dim = dim
    self.output_dim = n_levels * n_features_per_level

  def forward(self, x: torch.Tensor):
    return torch.cat([level(x) for level in self.levels], dim=-1)


class PositionalEncoding(EncodingLayer):

    @property
    def output_channels(self) -> int:
        return self.num_frequencies * self.domain_dim * 2

    def forward(self, x: T):
        shape = x.shape[:-1]
        if x.dim() != 2:
            x = x.view(-1, self.domain_dim)
        out: T = torch.einsum('f,nd->nfd', self.frequencies, x)
        out = torch.cat((torch.cos(out), torch.sin(out)), dim=2).view(-1, self.output_channels)
        return out.view(*shape, -1)

    def __init__(self, domain_dim: int, num_frequencies: int):
        super(PositionalEncoding, self).__init__()
        self.domain_dim = domain_dim
        self.num_frequencies = num_frequencies
        frequencies = torch.tensor([2. ** i * np.pi for i in range(num_frequencies)])
        self.register_buffer("frequencies", frequencies)


class RadialBasisEncoding(EncodingLayer):

    @property
    def output_channels(self) -> int:
        return self.num_frequencies

    def forward(self, x: T):
        shape = x.shape[:-1]
        if x.dim() != 2:
            x = x.view(-1, self.domain_dim)
        out = (x[:, None, :] - self.centres[None, :, :]).pow(2).sum(2)
        out = out * self.sigma[None, :] ** 2
        out = torch.exp(-out)
        return out.view(*shape, -1)

    def __init__(self, domain_dim: int, num_frequencies: int, std: int):
        super(RadialBasisEncoding, self).__init__()
        self.domain_dim = domain_dim
        self.num_frequencies = num_frequencies * 2
        centres = torch.rand(self.num_frequencies, domain_dim) * 2 - 1
        sigma = (torch.randn(self.num_frequencies).abs() * std + 1)
        sigma = sigma.sort()[0]
        self.register_buffer("centres", centres)
        self.register_buffer("sigma", sigma)


class PeriodicRadialBasisEncoding(EncodingLayer, abc.ABC):

    @property
    def output_channels(self) -> int:
        return 2 * self.num_frequencies

    def forward(self, x: T):
        shape = x.shape[:-1]
        if x.dim() != 2:
            x = x.view(-1, self.domain_dim)
        x_a = x[:, None, :] + self.offsets[None, :]  # n f d
        x_b = x_a + (1 / self.sigma[None, :, None])  # n f d
        out = torch.stack((x_a, x_b), dim=2)
        out = (out % (2 / self.sigma[None, :, None, None])) * 2 - (2 / self.sigma[None, :, None, None])
        out = out.pow(2).sum(3)  # n f 2
        out = out * self.sigma[None, :, None] ** 2
        out = out.view(-1, self.output_channels)
        out = torch.exp(-out) * 2 - 1
        return out.view(*shape, self.output_channels)

    @abc.abstractmethod
    def init_frequencies(self, std: float) -> T:
        raise NotImplemented

    def __init__(self, domain_dim: int, num_frequencies: int, std: float):
        super(PeriodicRadialBasisEncoding, self).__init__()
        self.domain_dim = domain_dim
        self.num_frequencies = num_frequencies
        sigma = self.init_frequencies(std)
        offsets = (torch.rand(self.num_frequencies, domain_dim) * 2 - 1) % (2 / sigma[:, None])
        sigma = sigma.sort()[0]
        self.register_buffer("offsets", offsets)
        self.register_buffer("sigma", sigma)


class RandomRadialBasisGridEncoding(PeriodicRadialBasisEncoding):

    def init_frequencies(self, std: float) -> T:
        return torch.randn(self.num_frequencies).abs() * std + 1


class UniformRadialBasisGridEncoding(PeriodicRadialBasisEncoding):

    def init_frequencies(self, std: float) -> T:
        frequencies = torch.linspace(0, std * np.sqrt(3), self.num_frequencies)
        frequencies = frequencies + frequencies[1] / 2
        return frequencies


class FFModel(EncodedMlpModel):

    def get_encoding_layer(self) -> EncodingLayer:
        return GaussianRandomFourierFeatures(self.opt.domain_dim, self.opt.num_frequencies, self.opt.std)


class PEModel(EncodedMlpModel):

    def get_encoding_layer(self) -> EncodingLayer:
        return PositionalEncoding(self.opt.domain_dim, self.opt.num_frequencies)


class RbfModel(EncodedMlpModel):

    def get_encoding_layer(self) -> EncodingLayer:
        return RadialBasisEncoding(self.opt.domain_dim, self.opt.num_frequencies, self.opt.std)


class PrbfModel(EncodedMlpModel):

    def get_encoding_layer(self) -> EncodingLayer:
        return UniformRadialBasisGridEncoding(self.opt.domain_dim, self.opt.num_frequencies, self.opt.std)

class HGModel(EncodedMlpModel):

    def get_encoding_layer(self) -> EncodingLayer:
        return Hashgrid(self.opt.domain_dim, self.opt.L, self.opt.F, self.opt.map_size, self.opt.base_res, self.opt.finest_res)

def get_model(params: ModelParams, model_type: EncodingType) -> EncodedMlpModel:
    if model_type is EncodingType.FF:
        model = FFModel(params)
    elif model_type is EncodingType.NoEnc:
        model = BaseModel(params)
    elif model_type == EncodingType.RBF:
        model = RbfModel(params)
    elif model_type == EncodingType.PRBF:
        model = PrbfModel(params)
    elif model_type is EncodingType.PE:
        model = PEModel(params)
    elif model_type is EncodingType.HG:
        model = HGModel(params)

    else:
        raise ValueError(f"{model_type.value} is not supported")
    return model


def mean_abs_weights(model):
    total_sum = 0.0
    total_num = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            total_sum += torch.abs(param.data).sum().item()
            total_num += param.numel()

    mean_val = total_sum / total_num if total_num > 0 else 0
    return mean_val


class MaskModel(nn.Module):
    def __init__(self, mask_model, cmlp, prob = torch.tensor([1]), lambda_cost=0.01, mask_act = lambda x: x, loss = nnf.mse_loss, threshold = 0, compensate_inv_prob = False, bn = False):
        super().__init__()
        self.is_progressive = True

        self.mask = mask_model
        self.cmlp = cmlp
        self.mask_act = mask_act
        self.loss = loss

        self.lambda_cost = lambda_cost
        self.encoding_dim = cmlp.encoding_dim
        self.device = next(self.mask.parameters()).device
        self.weight_tensor = (cmlp.model.encode.frequencies - threshold).to(self.device)
        self.best_model = None
        self.lowest_loss = 9999
        self.bn = bn

        self.batch_norm = nn.BatchNorm1d(cmlp.model.encode.frequencies.shape[-1], momentum=1).to(self.device)


        if not compensate_inv_prob:
            prob = torch.ones(1)
        inv_prob = (1. / prob).float().to(self.device)
        self.inv_prob = inv_prob / inv_prob.mean()

        wandb.config.update({'lambda_cost': self.lambda_cost, 'threshold': threshold})

    def fit(self, vs_in, labels, image, out_path, tag, num_iterations=1000, vs_base=None, lr = 1e-3, weight_decay = 1, eval_labels = None, log = lambda *args, **kwargs: None):
        wandb.config.update({'weight_decay': weight_decay})
        optimizer = OptimizerW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9998)

        logger = train_utils.Logger().start(num_iterations)
        vs_in, labels = vs_in.to(self.device), labels.to(self.device)
        for i in range(num_iterations):
            optimizer.zero_grad()

            mask_original = self.train_iter(vs_in, labels, logger)
            optimizer.step()
            scheduler.step()

            if i % 500 == 0 and vs_base is not None:
                log(self, image, out_path, tag, vs_base, self.device, i, labels = eval_labels)
                wandb.log({'main model weights size': mean_abs_weights(self.cmlp), 'mask model weights size': mean_abs_weights(self.mask)})
            logger.reset_iter()
        logger.stop()

        self.load_state_dict(self.best_model)

        return mask_original

    def train_iter(self, vs_in, labels, logger = None):
        result, mask_original = self.forward(vs_in, get_mask=True)

        mse_loss = self.loss(
                result, labels, reduction='none')
        mse_loss = mse_loss.mean(1) * self.inv_prob
        mse_loss = mse_loss.mean()

        # Multiply the mask with the weight tensor before calculating the cost
        weighted_mask = torch.abs(mask_original) * self.weight_tensor
        weighted_mask = weighted_mask.mean(1) * self.inv_prob

        mask_cost = self.lambda_cost * weighted_mask.mean()

        total_loss = mse_loss + mask_cost
        if total_loss < self.lowest_loss:
            self.lowest_loss = total_loss
            self.best_model = copy.deepcopy(self.state_dict())
            wandb.log({'lowest_loss': self.lowest_loss})

        if logger is not None:
            logger.stash_iter('mse_train', mse_loss)
            logger.stash_iter('mask_cost', mask_cost)
            logger.stash_iter('total_loss', total_loss)
        wandb.log({'mse_train': mse_loss, 'mask_cost': mask_cost, 'total_loss': total_loss})

        total_loss.backward()
        return mask_original

    def forward(self, vs_in, get_mask = False):
        mask_original = self.mask_act(self.mask(vs_in))
        if self.bn:
            mask_original = self.batch_norm(mask_original)
            # mask_original = self.layer_norm(mask_original)

        # mask = torch.stack([mask_original, mask_original], dim=2).view(vs_in.shape[0], -1)
        # ones = torch.ones((vs_in.shape[0], self.cmlp.model.model.model[0].in_features - mask.shape[1]), device=vs_in.device)
        # mask = torch.cat([ones, mask], dim=-1)
        out = self.cmlp(vs_in, override_mask=mask_original)

        if get_mask:
            return out, mask_original
        else:
            return out


