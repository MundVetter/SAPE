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
import wandb


def mask_to_hm(mask, out_shape, img_shape):
    if mask.dim() != len(out_shape):
        mask: T = mask.unsqueeze(0).expand(out_shape[0], mask.shape[0])
    hm = mask.sum(1) / mask[:, 4:].sum(1).max()
    hm = image_utils.to_heatmap(hm)
    hm = hm.view(*img_shape[:-1], 3)
    return hm



def plot_image(model: encoding_controler.EncodedController, vs_in: T, ref_image: ARRAY, batch_size=16*16):
    out, mask = model.predict(vs_in, batch_size=batch_size, return_mask = True)
    # remove the first 4 channels (the mask)
    mask = mask[:, 4:]
    hms = []
    hm = mask_to_hm(mask, out.shape, ref_image.shape)
    hms.append(hm)

    # create 4 new heatmaps each of a different section of the mask
    mask_split = mask.shape[1] // 4
    for i in range(4):
        mask = mask[:, i * mask_split: (i + 1) * mask_split]
        hm = mask_to_hm(mask, out.shape, ref_image.shape)
        hms.append(hm)

    out = out.view(ref_image.shape)
    model.train()
    return out, hms

def export_images(model, image, out_path, tag, vs_base, device, batch_size = 16*16, i = 0):
    with torch.no_grad():
        extra = 'mask'
        out, hm = plot_image(
                    model, vs_base.to(device), image, batch_size)
        files_utils.export_image(
                    out, out_path / f'opt_{tag}_{extra}' / f'{i:04d}.png')
        wandb.log({f'opt_{tag}_{extra}': wandb.Image(str(out_path / f'opt_{tag}_{extra}' / f'{i:04d}.png'))})
        if hm is not None:
            for i, hm in enumerate(hm):
                files_utils.export_image(
                            hm, out_path / f'heatmap_{tag}_{i}_{extra}' / f'{i:04d}.png')
                wandb.log({f'heatmap_{tag}_{extra}': wandb.Image(str(out_path / f'heatmap_{tag}_{i}_{extra}' / f'{i:04d}.png'))})

class MaskModel(nn.Module):
    def __init__(self, model, prob, lambda_cost=0.16, num_masks = 3, num_freqs = 32, mask_hidden_dim = 128, mask_layers = 2):
        super().__init__()
        self.model = model
        self.lambda_cost = lambda_cost
        self.encoding_dim = model.encoding_dim

        self.device = next(self.model.parameters()).device

        self.masks = nn.ModuleList()
        model_params = encoding_models.ModelParams(domain_dim = 4, num_layers = mask_layers, hidden_dim = mask_hidden_dim, output_channels = 1)
        self.masks.append(Mask(encoding_models.BaseModel(model_params)).to(self.device))

        for _ in range(num_masks - 1):
            model_params = encoding_models.ModelParams(use_id_encoding=True, num_frequencies = num_freqs, domain_dim = 4, num_layers = mask_layers, hidden_dim = mask_hidden_dim, output_channels = 1, std = 20)
            self.masks.append(Mask(encoding_models.MultiModel2(model_params)).to(self.device))

        inv_prob = (1. / prob).float().to(self.device)
        inv_prob = inv_prob / inv_prob.mean()
        self.inv_prob = inv_prob

        wandb.init(project="mund-thesis",
            config={
                "lambda_cost": self.lambda_cost,
                "num_masks": len(self.masks),
                "mask_hidden_dim": mask_hidden_dim,
                "mask_layers": mask_layers,
                "num_freqs_mask": num_freqs,
                "num_freqs_model": self.model.model.encode.frequencies.shape[0] // 2
            })

    def fit(self, vs_in, labels, image, out_path, tag, batch_size = 4*4, num_iterations=1000, vs_base=None, lr= 1e-3):
        optimizer = Optimizer(self.parameters(), lr=lr)
        wandb.config.update({"lr": lr, "batch_size": batch_size, "num_iterations": num_iterations})
        wandb.watch(self, log="all", log_freq=100)
        # Freeze the parameters of the frozen model
        logger = train_utils.Logger().start(num_iterations * math.ceil(vs_in.shape[0] / batch_size))
        vs_in, labels = vs_in.to(self.device), labels.to(self.device)
        for i in range(num_iterations):
            # indices = torch.randperm(vs_in.shape[0]).to(self.device)
            for b_idx in range(0, vs_in.shape[0], batch_size):
                optimizer.zero_grad()

                # selected_indices = indices[b_idx:min(vs_in.shape[0], b_idx+batch_size)]
                # vs_in_batch = vs_in[selected_indices]
                # labels_batch = labels[selected_indices]
                vs_in_batch = vs_in[b_idx:min(vs_in.shape[0], b_idx+batch_size)]
                labels_batch = labels[b_idx:min(vs_in.shape[0], b_idx+batch_size)]

                mask_loss, mask, out = self.forward(vs_in_batch)

                mse_loss = nnf.mse_loss(
                    out, labels_batch, reduction='none')
                # mse_loss = mse_loss.mean(1) * self.inv_prob
                mse_loss = mse_loss.mean()

                # Multiply the mask with the weight tensor before calculating the cost
                total_loss = mse_loss + mask_loss
                logger.stash_iter({'mse_train': mse_loss, "mask_loss": mask_loss, "total_loss": total_loss})
                wandb.log({'mse_train': mse_loss, "mask_loss": mask_loss, "total_loss": total_loss})

                total_loss.backward()
                optimizer.step()
                logger.reset_iter()
            if i % 100 == 0 and vs_base is not None:
                export_images(self, image, out_path, tag, vs_base, self.device, batch_size, i = i)
        logger.stop()
        wandb.finish()

        return mask

    def forward(self, vs_in):
        ones = torch.ones_like(vs_in, device = vs_in.device)
        loss_weights = [0.1, 0.2, 1.0]
        
        # List of frequencies for each mask, using next mask (or frozen_model for last mask)
        freqs = [mask.model.encode.encoders[0].frequencies for mask in self.masks[1:]] + [self.model.model.encode.frequencies]

        mask_original, mask = self.masks[0].forward(vs_in, frequencies=freqs[0])
        if len(self.masks) > 1:
            mask = torch.cat([ones, mask], dim=-1)
        mask_costs = [self.mask_loss(mask_original, freqs[0]) * loss_weights[0]]

        for i in range(1, len(self.masks)):
            mask = mask.repeat_interleave(freqs[i].shape[-1], dim=0)
            mask_original, mask = self.masks[i].forward(vs_in, frequencies=freqs[i], mask=mask)
            mask_costs.append(self.mask_loss(mask_original, freqs[i]) * loss_weights[i])

            if i < len(self.masks) - 1:
                mask = torch.cat([ones, mask], dim=-1)

        out = self.model(vs_in, override_mask=mask)

        return sum(mask_costs), mask, out
        
    def predict(self, vs_in, batch_size = 16*16, return_mask=False):
        self.eval()
        out_all = None
        mask_all = None
        with torch.no_grad():
            for b_idx in range(0, vs_in.shape[0], batch_size):
                vs_in_batch = vs_in[b_idx:min(vs_in.shape[0], b_idx+batch_size)]
                _, mask, out = self.forward(vs_in_batch)

                if out_all is None and mask_all is None:
                    out_all = out
                    mask_all = mask
                else:
                    out_all = torch.cat([out_all, out], dim=0)
                    mask_all = torch.cat([mask_all, mask], dim=0)
        self.train()
        if return_mask:
            return out_all, mask_all
        else:
            return out_all

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
        if (((i + 1) % freq == 0) or (i == 3)) and verbose:
            export_images(model, target_image, out_path, tag, vs_base, device, batch_size, i = i)
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
         IMAGE_PATH="images/chibi.jpg",
         ENCODING_TYPE = EncodingType.FF,
         CONTROLLER_TYPE = ControllerType.GlobalProgression) -> int:
    
    # wandb.init(mode="disabled")

    device = CUDA(0)
    image_path = constants.DATA_ROOT / IMAGE_PATH
    os.makedirs(constants.CHECKPOINTS_ROOT, exist_ok=True)
    print(device)
    name = files_utils.split_path(IMAGE_PATH)[1]

    scale = .5
    group = init_source_target(image_path, name, scale=scale,
                               max_res=64, square=False, non_uniform_sampling=NON_UNIFORM)
    vs_base, vs_in, labels, target_image, image_labels, (masked_cords, masked_labels, masked_image), prob = group

    model_params = encoding_models.ModelParams(domain_dim=2, output_channels=3, num_frequencies=128,
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

    mask_model_params = encoding_models.ModelParams(domain_dim=2, output_channels=256, num_frequencies=2,
                                                    hidden_dim=256, std=5., num_layers=3)
    weight_tensor = (model.model.encode.frequencies**2).sum(0)**0.5
    control_params_2 = encoding_controler.ControlParams(
        num_iterations=1000, epsilon=1e-5)

    if LEARN_MASK:
        optMask = MaskModel(model, prob, lambda_cost=0.16)
        mask = optMask.fit(vs_in, labels, target_image, out_path, tag, 512*4, EPOCHS,
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
