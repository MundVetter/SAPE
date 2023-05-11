import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'

from utils.image_utils import init_source_target
from custom_types import *
from models import encoding_controler, encoding_models
from utils import files_utils, train_utils, image_utils
import constants
import copy
import matplotlib.pyplot as plt
import os
from pathlib import Path
import math
import csv



def plot_image(model: encoding_controler.EncodedController, vs_in: T, ref_image: ARRAY):
    model.eval()
    with torch.no_grad():
        # if model.is_progressive or mask_model is not None:
        #     if mask_model is not None:
        #         _, mask = mask_model(vs_in)
        #         out = model(vs_in, override_mask=mask)
        #     else:
        #         out, mask = model(vs_in, get_mask=True)
        #     if mask.dim() != out.dim():
        #         mask: T = mask.unsqueeze(0).expand(out.shape[0], mask.shape[0])
        #     hm = mask.sum(1) / mask.shape[1]
        #     hm = image_utils.to_heatmap(hm)
        #     hm = hm.view(*ref_image.shape[:-1], 3)
        # else:
        out = model(vs_in)
        hm = None
        out = out.view(ref_image.shape)
    model.train()

    return out, hm


class MaskModel(nn.Module):
    def __init__(self, frozen_model, prob, lambda_cost=0.01):
        super().__init__()
        self.frozen_model = frozen_model
        self.lambda_cost = lambda_cost
        self.encoding_dim = frozen_model.encoding_dim

        self.device = next(self.frozen_model.parameters()).device
        
        model_params2 = encoding_models.ModelParams(domain_dim = 4, num_layers = 2, hidden_dim = 256, output_channels = 1)
        self.mask1 = Mask(encoding_models.BaseModel(model_params2)).to(self.device)
        model_params = encoding_models.ModelParams(use_id_encoding=True, num_frequencies = 16, domain_dim = 4, num_layers = 2, hidden_dim = 256, output_channels = 1)
        self.mask2 = Mask(encoding_models.MultiModel2(model_params)).to(self.device)


        inv_prob = (1. / prob).float().to(self.device)
        inv_prob = inv_prob / inv_prob.mean()
        self.inv_prob = inv_prob

    def fit(self, vs_in, labels, image, out_path, tag, num_iterations=1000, vs_base=None):
        optimizer = Optimizer(self.parameters(), lr=1e-4,)
        # Freeze the parameters of the frozen model
        logger = train_utils.Logger().start(num_iterations)
        vs_in, labels = vs_in.to(self.device), labels.to(self.device)
        for i in range(num_iterations):
            optimizer.zero_grad()

            mask_loss, mask, out = self.forward(vs_in)

            mse_loss = nnf.mse_loss(
                out, labels, reduction='none')
            mse_loss = mse_loss.mean(1) * self.inv_prob
            mse_loss = mse_loss.mean()

            # Multiply the mask with the weight tensor before calculating the cost
            total_loss = mse_loss + mask_loss
            logger.stash_iter({'mse_train': mse_loss, "mask_loss": mask_loss, "total_loss": total_loss})

            if i % 100 == 0 and vs_base is not None:
                export_images(self, image, out_path, tag, vs_base, self.device, i = i)

            total_loss.backward()
            optimizer.step()
            logger.reset_iter()
        logger.stop()

        return mask

    def forward(self, vs_in):
        freq1 = self.mask2.model.encode.encoders[0].frequencies
        mask_original, mask = self.mask1.forward(vs_in, frequencies=freq1)
        ones = torch.ones_like(vs_in, device = vs_in.device)
        mask = torch.cat([ones, mask], dim=-1)
        mask = mask.repeat_interleave(self.frozen_model.model.encode.frequencies.shape[-1], dim=0)
        freq2 = self.frozen_model.model.encode.frequencies
        mask_original2, mask2 = self.mask2.forward(vs_in, frequencies = freq2, mask=mask)

        out = self.frozen_model(vs_in, override_mask=mask2)

        mask_cost = self.mask_loss(mask_original, freq1)
        mask_cost2 = self.mask_loss(mask_original2, freq2)

        if self.training:
            return mask_cost + mask_cost2, mask, out
        else:
            return out

        # check if model is progressive
        # if self.frozen_model.is_progressive:
        #     mask = torch.stack([mask_original, mask_original], dim=2).view(-1, self.encoding_dim - 2)
        #     ones = torch.ones_like(vs_in, device=vs_in.device)
        #     mask = torch.cat([ones, mask], dim=-1)
        # else:
        #     mask = torch.stack([mask_original, mask_original], dim=2).view(-1, self.encoding_dim)
    
    def mask_loss(self, mask, freq):
        return self.lambda_cost * (mask * (freq**2).sum(0)**0.5).mean()
    
class Mask(nn.Module):
    def __init__(self, model, sigma_freq=5):
        super().__init__()
        self.model = model
        self.sigma_freq = sigma_freq

    def forward(self, vs_in, frequencies, mask=None):
        freq = frequencies / (self.sigma_freq * 3)

        # Repeat freq to match the number of rows in vs_in
        freq_repeated = freq.repeat(vs_in.shape[0], 1).reshape(-1, 2)

        # Repeat vs_in to match the size of freq
        vs_in_repeated = vs_in.repeat_interleave(freq.shape[-1], dim=0)

        # Concatenate vs_in and freq along the given dimension
        merged = torch.cat((vs_in_repeated, freq_repeated), dim=1)

        mask_original = self.model(merged, override_mask=mask).reshape(shape=(-1, freq.shape[-1]))
        mask_original = torch.sigmoid(mask_original)


        mask = torch.stack([mask_original, mask_original], dim=2).view(-1, freq.shape[-1] * 2)
        ones = torch.ones_like(vs_in, device = vs_in.device)
        mask = torch.cat([ones, mask], dim=-1)

        return mask_original, mask
    
def export_images(model, image, out_path, tag, vs_base, device, mask_model=None, i = 0):
    with torch.no_grad():
        extra = 'mask' if mask_model is not None else 'no_mask'
        out, hm = plot_image(
                    model, vs_base.to(device), image)
        files_utils.export_image(
                    out, out_path / f'opt_{tag}_{extra}' / f'{i:04d}.png')
        if hm is not None:
            files_utils.export_image(
                        hm, out_path / f'heatmap_{tag}_{extra}' / f'{i:04d}.png')


def optimize(encoding_type: EncodingType, model_params,
             controller_type: ControllerType, control_params: encoding_controler.ControlParams, group, tag, out_path, device: D,
             freq: int, verbose=False, mask=None, model=None, mask_model=None, lr=1e-3):
    vs_base, vs_in, labels, target_image, image_labels, _, prob = group
    if model is None:
        model = encoding_controler.get_controlled_model(
            model_params, encoding_type, control_params, controller_type).to(device)
    block_iterations = model.block_iterations
    vs_base, vs_in, labels, image_labels = vs_base.to(device), vs_in.to(
        device), labels.to(device), image_labels.to(device)
    inv_prob = (1. / prob).float().to(device)
    inv_prob = inv_prob / inv_prob.mean()
    opt = Optimizer(model.parameters(), lr=lr)
    logger = train_utils.Logger().start(control_params.num_iterations, tag=tag)
    files_utils.export_image(target_image, out_path / 'target.png')
    for i in range(control_params.num_iterations):
        opt.zero_grad()
        if mask is None:
            out = model(vs_in)
        else:
            out = model(vs_in, override_mask=mask)
        loss_all = nnf.mse_loss(out, labels, reduction='none')
        loss_all[:, 0] *= inv_prob
        loss_all[:, 1] *= inv_prob
        loss_all[:, 2] *= inv_prob
        loss = loss_all.mean()
        if i == 0:
            print(loss)
        loss.backward()
        opt.step()
        model.stash_iteration(loss_all.mean(-1))
        logger.stash_iter('mse_train', loss)
        if block_iterations > 0 and (i + 1) % block_iterations == 0:
            model.update_progress()
        if (((i + 1) % freq == 0) or (i == 0)) and verbose:
            export_images(model, target_image, out_path, tag, vs_base, device, i = i, mask_model=mask_model)
        logger.reset_iter()
    logger.stop()

    return model

def psnr(img1, img2, **_):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def rgb_to_grayscale(img):
    # The weights correspond to the conversion formula: 
    # Y = 0.2989 * R + 0.5870 * G + 0.1140 * B
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, -1, 1, 1).to(img.device)
    return (img * weights).sum(dim=1, keepdim=True)

def ssim(img1, img2, img_shape=None, window_size=11, k1=0.01, k2=0.03, L=1.0, **_):
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = torch.ones((1, 1, window_size, window_size)) / (window_size ** 2)
    
    img1 = img1.view(1, img1.shape[1], img_shape[0], -1)
    img2 = img2.view(1, img2.shape[1], img_shape[0], -1)

    img1_gray = rgb_to_grayscale(img1)
    img2_gray = rgb_to_grayscale(img2)

    window = window.to(img1_gray.device)

    mu1 = nnf.conv2d(img1_gray, window, padding=window_size // 2, groups=1)
    mu2 = nnf.conv2d(img2_gray, window, padding=window_size // 2, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = nnf.conv2d(img1_gray * img1_gray, window, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = nnf.conv2d(img2_gray * img2_gray, window, padding=window_size // 2, groups=1) - mu2_sq
    sigma12 = nnf.conv2d(img1_gray * img2_gray, window, padding=window_size // 2, groups=1) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return torch.mean(ssim_map)


def evaluate(model, vs_in, labels, funcs = [], mask=None, **kwargs):
    model.eval()
    with torch.no_grad():
        if type(funcs) is list:
            out = model(vs_in, override_mask=mask)
            return [func(out, labels, **kwargs) for func in funcs]
        else:
            out =  model(vs_in, override_mask=mask)
            return funcs(out, labels, **kwargs)

def mean_of_groups(tensor, num_groups):
    group_size = len(tensor) // num_groups
    if group_size == 0:
        raise ValueError("The number of groups should be less than or equal to the tensor length.")

    output = torch.zeros(num_groups)
    for i in range(num_groups):
        start = i * group_size
        end = start + group_size
        output[i] = tensor[start:end].mean()

    return output

def evaluate_configurations(model, vs_in, labels, funcs, device, name="", **kwargs):
    results = {}
    vs_in = vs_in.to(device)
    labels = labels.to(device)
    results[f"{name}_model_no_mask"] = evaluate(model, vs_in, labels, funcs, **kwargs)
    return results

def pretty_print_results(results, name, funcs):
    print("=====================================")
    print(f"Results for {name}")
    if type(funcs) is not list:
        for key, value in results.items():
            print(f"{key} [{funcs.__name__}]: {value}")
    else:
        for key, value in results.items():
            for i, func in enumerate(funcs):
                print(f"{key} [{func.__name__}]: {value[i]}")
    print("=====================================")


def save_results_to_csv(results, name, funcs, path, tag):
    file_name = path / f"{name}-{tag}-results.csv"
    
    with open(file_name, mode='w', newline='') as csv_file:
        fieldnames = ['configuration', 'function', 'value']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for key, value in results.items():
            if type(funcs) is not list:
                writer.writerow({'configuration': key, 'function': funcs.__name__, 'value': float(value)})
            else:
                for i, func in enumerate(funcs):
                    writer.writerow({'configuration': key, 'function': func.__name__, 'value': float(value[i])})

def main(PRETRAIN=True,
         LEARN_MASK=True,
         RETRAIN=False,
         NON_UNIFORM=False,
         EPOCHS=4000,
         IMAGE_PATH="natural_images/image_000.jpg",
         ENCODING_TYPE = EncodingType.FF,
         CONTROLLER_TYPE = ControllerType.GlobalProgression) -> int:
    device = CUDA(0)
    image_path = constants.DATA_ROOT / IMAGE_PATH
    os.makedirs(constants.CHECKPOINTS_ROOT, exist_ok=True)
    print(device)
    name = files_utils.split_path(IMAGE_PATH)[1]

    scale = .25
    group = init_source_target(image_path, name, scale=scale,
                               max_res=512, square=False, non_uniform_sampling=NON_UNIFORM)
    vs_base, vs_in, labels, target_image, image_labels, (masked_cords, masked_labels, masked_image), prob = group

    model_params = encoding_models.ModelParams(domain_dim=2, output_channels=3, num_frequencies=16,
                                               hidden_dim=256, std=20., num_layers=3)
    control_params = encoding_controler.ControlParams(
        num_iterations=1, epsilon=1e-3, res=128)

    tag = f'{name}_{ENCODING_TYPE.value}_{CONTROLLER_TYPE.value}_{NON_UNIFORM}'
    out_path = constants.CHECKPOINTS_ROOT / '2d_images' / name
    os.makedirs(out_path, exist_ok=True)

    if PRETRAIN:
        model = optimize(ENCODING_TYPE, model_params, CONTROLLER_TYPE, control_params, group, tag, out_path, device,
                         50, verbose=True)
        torch.save(model.state_dict(), out_path / f'model_{tag}.pt')
    else:
        model = encoding_controler.get_controlled_model(
            model_params, ENCODING_TYPE, control_params, CONTROLLER_TYPE).to(device)
        model.load_state_dict(torch.load(out_path / f'model_{tag}.pt'))

    mask_model_params = encoding_models.ModelParams(domain_dim=2, output_channels=256, num_frequencies=256,
                                                    hidden_dim=256, std=5., num_layers=3)
    weight_tensor = (model.model.encode.frequencies**2).sum(0)**0.5
    control_params_2 = encoding_controler.ControlParams(
        num_iterations=1000, epsilon=1e-5)

    if LEARN_MASK:
        optMask = MaskModel(model, prob, lambda_cost=0.16)
        mask = optMask.fit(vs_in, labels, target_image, out_path, tag, EPOCHS,
                           vs_base=vs_base).detach()

        torch.save(mask, out_path / 'mask.pt')
        torch.save(optMask.state_dict(), out_path / f'mask_model_{tag}.pt')
    else:
        mask = torch.load(out_path / 'mask.pt')
        mask_model = encoding_controler.get_controlled_model(
            mask_model_params, ENCODING_TYPE, control_params_2, ControllerType.NoControl).to(device)
        mask_model.load_state_dict(torch.load(
            out_path / f'mask_model_{tag}.pt'))
        optMask = MaskModel(mask_model, model, weight_tensor, prob, lambda_cost=0.16)

    # if RETRAIN:
    #     # only retrain last layer
    #     for param in model.parameters():
    #         param.requires_grad = False
    #     for param in model.model.model.model[-3:].parameters():
    #         param.requires_grad = True
    #     model2 = optimize(ENCODING_TYPE, model_params, CONTROLLER_TYPE, control_params, group, tag, out_path, device,
    #                       50, verbose=True, mask=mask, model=model, mask_model=optMask, lr=1e-4)
    #     torch.save(model2.state_dict(), out_path / f'model2_{tag}.pt')
    # else:
    #     model2 = encoding_controler.get_controlled_model(
    #         model_params, ENCODING_TYPE, control_params, CONTROLLER_TYPE).to(device)
    #     model2.load_state_dict(torch.load(out_path / f'model2_{tag}.pt'))

    # Evaluation
    res_train = evaluate_configurations(model, vs_in, labels, psnr, device, "train")
    res_test = evaluate_configurations(model, vs_base, image_labels, [psnr, ssim], device, "test", img_shape = target_image.shape)
    res_masked = evaluate_configurations(model, masked_cords, masked_labels, psnr, device, "test_masked")

    pretty_print_results(res_train, "train", psnr)
    pretty_print_results(res_test, "test", [psnr, ssim])
    pretty_print_results(res_masked, "test_masked", psnr)

    tag_without_filename = f"{ENCODING_TYPE.value}_{CONTROLLER_TYPE.value}_{NON_UNIFORM}"
    save_results_to_csv(res_train, "train", psnr, out_path, tag_without_filename)
    save_results_to_csv(res_test, "test", [psnr, ssim], out_path, tag_without_filename)
    save_results_to_csv(res_masked, "test_masked", psnr, out_path, tag_without_filename)

    return 0


if __name__ == '__main__':
    exit(main())
