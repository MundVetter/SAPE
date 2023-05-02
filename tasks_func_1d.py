from custom_types import *
from models import encoding_models, encoding_controler
from utils import files_utils, train_utils, image_utils
import constants
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import imageio
import matplotlib.pyplot as plt
import copy

class Function:
    def __call__(self, x: T) -> T:
        return self.function(x)

    @property
    def name(self):
        return self.function.__name__

    def __init__(self, function: Callable[[T], T]):
        self.function = function
        self.samples = None


def export_poly(vs: Optional[T], dots: Optional[T], path: str, colors: Optional[TNS] = None, r: int = 6,
                opacity: bool = False):

    res = 1000

    def to_rgba(c: T):
        c = (c * 255).long().tolist() + [0]
        c = tuple(c)
        return c

    points = []
    for pts in (vs, dots):
        if pts is not None:
            pts = pts.clone()
            pts = pts.clone()
            pts[:, 0] = (pts[:, 0] + 1) / 2
            pts[:, 1] = 1 - (pts[:, 1] + 1) / 2
            pts = (pts * (res - 200) + 100).long().tolist()
            pts = [tuple(xy) for xy in pts]
        points.append(pts)

    image = Image.new("RGB", (res, res), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    if vs is not None:
        if colors is None or colors[0] is None:
            draw.line(points[0], fill='black', width=2)
        else:
            for i in range(len(points[0]) - 1):
                fill = to_rgba(colors[0][i])
                line = points[0][i: i + 2]
                draw.line(line, fill=fill, width=1)
    if dots is not None:
        for i, (x, y) in enumerate(points[1]):
            if colors is not None and colors[1] is not None:
                fill = to_rgba(colors[1][i])
                outline = (80, 80, 80)
            else:
                fill = (255, 0, 0, 0)
                outline = None
            draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, width=1, outline=outline)

    path = files_utils.add_suffix(path, '.png')
    files_utils.init_folders(path)
    # if not constants.DEBUG:
    if opacity:
        image = V(image)
        mask = np.sum(image, axis=2) == 255 * 3
        alpha = np.ones((*image.shape[:-1], 1), dtype=image.dtype) * 255
        alpha[mask] = 0
        image = np.concatenate((image, alpha), axis=2)
        imageio.imsave(path, image)
    else:
        image.save(path)


def init_source_target(func: Function, num_samples: int):
    vs_base = torch.linspace(-1, 1, 3000).unsqueeze(-1)
    target_func = func(vs_base)
    target_func = torch.cat((vs_base, target_func), dim=1)
    if func.samples is None:
        vs_in = torch.linspace(0.5, 1, 14).unsqueeze(-1)
        vs_in = torch.cat((vs_in, torch.linspace(-1, 0.5, 12).unsqueeze(-1)), dim=0)
        vs_in += (vs_in[1, 0] - vs_in[0, 0]) / 2
        vs_in[-1] = .99
        labels = func(vs_in)
    else:
        vs_in = func.samples[:, 0].unsqueeze(-1)
        labels = func.samples[:, 1].unsqueeze(-1)
    return vs_base, vs_in, labels, target_func

def psine(x: T):
    a,  c, b = .01, 4.5, 4

    def f_(x_):
        return a * (torch.clamp(x_, 0) * 2 + b) ** c

    return .5 * torch.sin(f_(x))


class MaskOptimizer(nn.Module):
    def __init__(self, mask_model, frozen_model, weight_tensor,lambda_cost=0.01):
        super().__init__()
        self.mask_model = mask_model
        self.frozen_model = frozen_model
        # self.optimizer2 = Optimizer(self.frozen_model.parameters(), lr=mask_lr)
        self.lambda_cost = lambda_cost
        self.encoding_dim  = frozen_model.encoding_dim
        self.weight_tensor = weight_tensor
        self.num_freq = 128
        self.encode = encoding_models.GaussianRandomFourierFeatures(1, 128, 1.5).cuda()
        # self.encode2 = encoding_models.GaussianRandomFourierFeatures(1, 128, 2).cuda()

    def optimize_mask(self, vs_in, labels, num_iterations=1000):
        # Freeze the parameters of the frozen model
        optimizer = Optimizer(list(self.mask_model.parameters()) + list(self.frozen_model.parameters()), lr=0.001)
        # for param in self.frozen_model.parameters():
        #     param.requires_grad = False

        for i in range(num_iterations):
            optimizer.zero_grad()
            # self.optimizer2.zero_grad()

            mask_original, mask = self.forward(vs_in)
            frozen_model_output = self.frozen_model(vs_in, override_mask=mask)

            mse_loss = nnf.mse_loss(frozen_model_output, labels)

            # Multiply the mask with the weight tensor before calculating the cost
            weighted_mask = mask_original * self.weight_tensor
            mask_cost = self.lambda_cost * weighted_mask.mean()
            total_loss = mse_loss + mask_cost

            if i % 100 == 0:
                print(f'Iteration {i} - Loss: {total_loss.item()}')

            total_loss.backward()
            optimizer.step()
            # self.optimizer2.step()

        # unfreeze the parameters of the frozen model
        for param in self.frozen_model.parameters():
            param.requires_grad = True
        return mask


class Mask(nn.Module):
    def __init__(self, frozen_model, mask_model, encoding_dim, num_freq):
        super().__init__()
        self.mask_model = mask_model
        self.frozen_model = frozen_model
        self.encoding_dim = encoding_dim
        self.num_freq = num_freq
        self.encode = encoding_models.GaussianRandomFourierFeatures(1, 128, 1.5).cuda()
        # self.encode2 = encoding_models.GaussianRandomFourierFeatures(1, 128, 2).cuda()
    def forward(self, vs_in):
        # vs_in_encoded = self.encode(vs_in)
        vs_in_encoded = vs_in
        frequencies = self.frozen_model.model.encode.frequencies.squeeze() / 5.
        # frequencies_encoded = self.encode2(torch.unsqueeze(frequencies, 1))
        frequencies_encoded = frequencies
        # vs_repeated = vs_in_encoded.view(-1, self.num_freq * 2, 1).repeat(1, 1, frequencies.shape[0])
        vs_repeated = vs_in_encoded.view(-1, 1, 1).repeat(1, 1, frequencies.shape[0])
        # frequencies_repeated = frequencies_encoded.view(1, self.num_freq * 2, -1).repeat(vs_in_encoded.shape[0], 1, 1)
        frequencies_repeated = frequencies_encoded.view(1, 1, -1).repeat(vs_in_encoded.shape[0], 1, 1)
        inputs = torch.cat([vs_repeated, frequencies_repeated], dim=1)
        mask_original = self.mask_model(inputs)
        # mask_original = nnf.threshold(mask_original, 0.001, 0)
        # mask_original = torch.threshold(torch.sigmoid(self.mask_model(vs_in)), 0.01, 0)
        mask = torch.stack([mask_original, mask_original], dim=2).view(-1, self.encoding_dim - 1)
        ones = torch.ones_like(vs_in, device = vs_in.device)
        mask = torch.cat([ones, mask], dim=-1)
        return mask_original, mask



def optimize(func: Function, encoding_type: EncodingType, model_params,
             controller_type: ControllerType, control_params: encoding_controler.ControlParams,
             num_samples: int, device: D,
             freq=500, verbose=False, mask = None, model = None):
    name = func.name
    vs_base, vs_in, labels, target_func = init_source_target(func, num_samples)
    vs_base, vs_in, labels, target_func = vs_base.to(device), vs_in.to(device), labels.to(device), target_func.to(
        device)
    lr = 1e-5
    if model is None:
        model = encoding_controler.get_controlled_model(model_params, encoding_type, control_params, controller_type).to(device)

    tag = f'{encoding_type.value}_{controller_type.value}'
    if encoding_type is EncodingType.NoEnc:
        lr = 1e-4
    block_iterations = model.block_iterations
    out_path = f'{constants.CHECKPOINTS_ROOT}/1d/{name}/'
    opt = Optimizer(model.parameters(), lr=lr)
    logger = train_utils.Logger().start(control_params.num_iterations, tag=tag)
    export_poly(target_func, torch.cat((vs_in, labels), dim=1), f'{out_path}target.png', opacity=True)
    for i in range(control_params.num_iterations):
        opt.zero_grad()
        if mask is None:
            out = model(vs_in)
        else:
            out = model(vs_in, override_mask=mask)

        loss_all = nnf.mse_loss(out, labels, reduction='none')
        loss = loss_all.mean()
        loss.backward()
        opt.step()
        model.stash_iteration(loss_all.squeeze(-1))
        if block_iterations > 0 and (i + 1) % block_iterations == 0:
            model.update_progress()
        logger.reset_iter('loss', loss)
        if verbose and ((i + 1) % freq == 0 or i == 0):
            out = model(vs_base)
            aprox_func = torch.cat((vs_base, out), dim=1)
            export_poly(aprox_func, torch.cat((vs_in, labels), dim=1), f'{out_path}opt_{tag}/{i:05d}.png')
            if model.is_progressive:
                _, mask_base = model(vs_in, get_mask=True)
                if mask_base.dim() == 1:
                    mask_base = mask_base.unsqueeze(0).expand(vs_in.shape[0], mask_base.shape[0])
                hm_base = mask_base.sum(1) / mask_base.shape[1]
                hm_base = image_utils.to_heatmap(hm_base)
                export_poly(aprox_func, torch.cat((vs_in, labels), dim=1), f'{out_path}heatmap_{tag}/{i:05d}.png',
                            colors=(None, hm_base))
    logger.stop()
    files_utils.save_model(model, f'{out_path}model_{tag}.pth')
    if verbose and model.is_progressive:
        image_utils.gifed(f'{out_path}heatmap_{tag}/', .03, tag, reverse=False)
        files_utils.delete_all(f'{out_path}heatmap_{tag}/', '.png',
                               filter_out=lambda x: f'{control_params.num_iterations - 1}' == x[1])
    if verbose:
        image_utils.gifed(f'{out_path}opt_{tag}/', .03, tag, reverse=False)
        # files_utils.delete_all(f'{out_path}opt_{tag}/', '.png',
                            #    filter_out=lambda x: f'{control_params.num_iterations - 1}' == x[1])
    return model, vs_base, vs_in, labels

class CNN1x1(nn.Module):
    def __init__(self, input_channels, hidden_size):
        super(CNN1x1, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, hidden_size, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv_hidden = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_size, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        # x = self.conv_hidden(x)
        # x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x

def main() -> int:
    device = CUDA(0)
    encoding_type = EncodingType.FF
    controller_type = ControllerType.GlobalProgression
    func = Function(psine)
    num_samples = 25
    control_params = encoding_controler.ControlParams(num_iterations=1, epsilon=1e-5, res=num_samples//2)
    model_params = encoding_models.ModelParams(domain_dim=1, output_channels=1, num_frequencies=256,
                                               hidden_dim=256, std=5., num_layers=2)

    model, vs_base, vs_in, labels = optimize(func, encoding_type, model_params, controller_type, control_params, num_samples, device, freq=500, verbose=True)
    model_copy = copy.deepcopy(model)

    control_params_2 = encoding_controler.ControlParams(num_iterations=1, epsilon=1e-5, res=num_samples//2)
    mask_model_params = encoding_models.ModelParams(domain_dim=1, output_channels=128, num_frequencies=128, hidden_dim=32, std=5., num_layers=2)
    mask_model = CNN1x1(1+1, 128).to(device)
    weight_tensor = model.model.encode.frequencies.abs()

    # weight_tensor = torch.log(model.model.encode.frequencies.abs()*100)
    # insert zero weight
    # weight_tensor = torch.cat( device=device), weight_tensor), dim=1)
    mOpt = MaskOptimizer(mask_model, model, weight_tensor, lambda_cost=0.16)
    mask_2 = mOpt.optimize_mask(vs_in, labels, 4000).detach()

    model_2, vs_base, vs_in, labels = optimize(func, encoding_type, model_params, controller_type, control_params, num_samples, device, freq=500, verbose=True, mask=mask_2, model=model_copy)
    # base_mask = mask_model(vs_base)
    # # base_mask = torch.threshold(torch.sigmoid(mask_model(vs_base)), 0.01, 0)
    # mask = torch.stack([base_mask, base_mask], dim=2).view(-1, 256)
    # ones = torch.ones_like(vs_base, device = vs_base.device)
    # mask = torch.cat([ones, mask], dim=-1) 
    _, mask = mOpt(vs_base)

    pred2 = model_2(vs_base, override_mask=mask).detach().cpu().numpy()
    pred = model(vs_base, override_mask=mask).detach().cpu().numpy()
    pred_unmasked = model(vs_base).detach().cpu().numpy()


    x = vs_base.cpu().numpy()
    x1 = vs_in.cpu().numpy()
    plt.plot(x1, labels.cpu().numpy(), 'o', label='target')
    plt.plot(x, pred, label='pred mask')
    # plt.plot(x, pred2, label='pred mask2')
    # plt.plot(x, pred_unmasked, label='pred unmasked')
    plt.legend()
    plt.show()

    plt.plot(x, mask.detach().cpu().numpy().sum(1), label='mask')
    plt.show()
    
    plt.imshow(mask[:, 2:256].detach().cpu().numpy() / mask[:, 2:256].detach().cpu().numpy().max(1)[0])
    plt.show()

    return 0


if __name__ == '__main__':
    exit(main())
