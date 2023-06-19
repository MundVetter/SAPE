import wandb
from custom_types import *
import abc

from custom_types import OptimizerW, nn, nnf, torch
from utils import train_utils
import copy


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
        self.weight_tensor = (cmlp.model.encode.frequencies**2).sum(0)**0.5 - threshold
        self.device = next(self.mask.parameters()).device
        self.best_model = None
        self.lowest_loss = 9999
        self.bn = bn

        self.batch_norm = nn.BatchNorm1d(cmlp.model.encode.frequencies.shape[1], momentum=1).to(self.device)


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

            if i % 100 == 0 and vs_base is not None:
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

        mask = torch.stack([mask_original, mask_original], dim=2).view(vs_in.shape[0], -1)
        ones = torch.ones((vs_in.shape[0], self.cmlp.model.model.model[0].in_features - mask.shape[1]), device=vs_in.device)
        mask = torch.cat([ones, mask], dim=-1)
        out = self.cmlp(vs_in, override_mask=mask)

        if get_mask:
            return out, mask_original
        else:
            return out


def evaluate(model, vs_in, labels, func = [], **kwargs):
    model.eval()
    with torch.no_grad():
        out = model(vs_in)
        return func(out, labels, **kwargs)