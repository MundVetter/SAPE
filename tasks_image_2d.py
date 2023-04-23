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

def plot_image(model: encoding_controler.EncodedController, vs_in: T, ref_image: ARRAY, mask_model):
    model.eval()
    with torch.no_grad():
        if model.is_progressive or mask_model is not None:
            if mask_model is not None:
                _, mask = mask_model(vs_in)
                out = model(vs_in, override_mask=mask)
            else:
                out, mask = model(vs_in, get_mask=True)
            if mask.dim() != out.dim():
                mask: T = mask.unsqueeze(0).expand(out.shape[0], mask.shape[0])
            hm = mask.sum(1) / mask.shape[1]
            hm = image_utils.to_heatmap(hm)
            hm = hm.view(*ref_image.shape[:-1], 3)
        else:
            out = model(vs_in, get_mask=True)
            hm = None
        out = out.view(ref_image.shape)
    model.train()

    return out, hm


class MaskModel(nn.Module):
    def __init__(self, mask_model, frozen_model, weight_tensor, mask_lr=1e-4, lambda_cost=0.01):
        super().__init__()
        self.mask_model = mask_model
        self.frozen_model = frozen_model
        self.optimizer = Optimizer(self.mask_model.parameters(), lr=mask_lr)
        self.lambda_cost = lambda_cost
        self.encoding_dim = frozen_model.encoding_dim
        self.weight_tensor = weight_tensor
        self.device = next(self.mask_model.parameters()).device

    def fit(self, vs_in, labels, image, out_path, tag, num_iterations=1000, vs_base=None):
        # Freeze the parameters of the frozen model
        for param in self.frozen_model.parameters():
            param.requires_grad = False

        logger = train_utils.Logger().start(num_iterations)
        vs_in, labels = vs_in.to(self.device), labels.to(self.device)
        for i in range(num_iterations):
            self.optimizer.zero_grad()

            mask_original, mask = self.forward(vs_in)
            frozen_model_output = self.frozen_model(vs_in, override_mask=mask)

            mse_loss = nnf.mse_loss(
                frozen_model_output, labels, reduction='none')
            mse_loss = mse_loss.mean()

            # Multiply the mask with the weight tensor before calculating the cost
            weighted_mask = mask_original * self.weight_tensor
            mask_cost = self.lambda_cost * weighted_mask.mean()
            total_loss = mse_loss + mask_cost
            logger.stash_iter('mse_train', mse_loss)

            if i % 100 == 0 and vs_base is not None:
                export_images(self.frozen_model, image, out_path, tag, vs_base, self.device, self, i)

            total_loss.backward()
            self.optimizer.step()
            logger.reset_iter()
        logger.stop()

        # unfreeze the parameters of the frozen model
        for param in self.frozen_model.parameters():
            param.requires_grad = True
        return mask

    def forward(self, vs_in):
        mask_original = torch.sigmoid(self.mask_model(vs_in))
        mask = torch.stack([mask_original, mask_original],
                           dim=2).view(-1, self.encoding_dim - 2)
        ones = torch.ones_like(vs_in, device=vs_in.device)
        mask = torch.cat([ones, mask], dim=-1)
        return mask_original, mask
    
def export_images(model, image, out_path, tag, vs_base, device, mask_model=None, i = 0):
    with torch.no_grad():
        extra = 'mask' if mask_model is not None else 'no_mask'
        out, hm = plot_image(
                    model, vs_base.to(device), image, mask_model)
        files_utils.export_image(
                    out, out_path / f'opt_{tag}_{extra}' / f'{i:04d}.png')
        if hm is not None:
            files_utils.export_image(
                        hm, out_path / f'heatmap_{tag}_{extra}' / f'{i:04d}.png')


def optimize(encoding_type: EncodingType, model_params,
             controller_type: ControllerType, control_params: encoding_controler.ControlParams, group, tag, out_path, device: D,
             freq: int, verbose=False, mask=None, model=None, mask_model=None, lr=1e-3):
    vs_base, vs_in, labels, target_image, image_labels, _ = group
    if model is None:
        model = encoding_controler.get_controlled_model(
            model_params, encoding_type, control_params, controller_type).to(device)
    block_iterations = model.block_iterations
    vs_base, vs_in, labels, image_labels = vs_base.to(device), vs_in.to(
        device), labels.to(device), image_labels.to(device)
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

    # if verbose:
    #     image_utils.gifed(f'{out_path}opt_{tag}/', .07, tag, reverse=False)
    #     if model.is_progressive:
    #         image_utils.gifed(f'{out_path}heatmap_{tag}/', .07, tag, reverse=False)
    # files_utils.delete_all(f'{out_path}heatmap_{tag}/', '.png',
    # filter_out=lambda x: f'{control_params.#num_iterations - 1}' == x[1])
    # files_utils.delete_all(f'{out_path}opt_{tag}/', '.png',
    #  filter_out=lambda x: f'{control_params.#num_iterations - 1}' == x[1])
    return model

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def rgb_to_grayscale(img):
    # The weights correspond to the conversion formula: 
    # Y = 0.2989 * R + 0.5870 * G + 0.1140 * B
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, -1, 1, 1).to(img.device)
    return (img * weights).sum(dim=1, keepdim=True)

def ssim(img1, img2, img_shape, window_size=11, k1=0.01, k2=0.03, L=1.0):
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


def evaluate(model, vs_in, labels, img_shape, mask=None):
    model.eval()
    with torch.no_grad():
        out = model(vs_in, override_mask=mask)
        return psnr(out, labels), ssim(out, labels, img_shape)


def main(PRETRAIN=False,
         LEARN_MASK=False,
         RETRAIN=False,
         IMAGE_PATH="images/snow.jpg",
         ENCODING_TYPE = EncodingType.FF,
         CONTROLLER_TYPE = ControllerType.GlobalProgression) -> int:
    device = CUDA(0)
    image_path = constants.DATA_ROOT / IMAGE_PATH
    os.makedirs(constants.CHECKPOINTS_ROOT, exist_ok=True)
    print(device)
    name = files_utils.split_path(IMAGE_PATH)[1]
    scale = .25
    group = init_source_target(image_path, name, scale=scale,
                               max_res=512, square=False, non_uniform_sampling=False)
    vs_base, vs_in, labels, target_image, image_labels, _ = group

    model_params = encoding_models.ModelParams(domain_dim=2, output_channels=3, num_frequencies=256,
                                               hidden_dim=256, std=20., num_layers=3)
    control_params = encoding_controler.ControlParams(
        num_iterations=3000, epsilon=1e-3, res=128)

    name = files_utils.split_path(image_path)[1]
    tag = f'{name}_{ENCODING_TYPE.value}_{CONTROLLER_TYPE.value}'
    out_path = constants.CHECKPOINTS_ROOT / '2d_images' / name

    if PRETRAIN:
        model = optimize(ENCODING_TYPE, model_params, CONTROLLER_TYPE, control_params, group, tag, out_path, device,
                         50, verbose=True)
        torch.save(model.state_dict(), out_path / f'model_{tag}.pt')
    else:
        model = encoding_controler.get_controlled_model(
            model_params, ENCODING_TYPE, control_params, CONTROLLER_TYPE).to(device)
        model.load_state_dict(torch.load(out_path / f'model_{tag}.pt'))

    # model_copy = copy.deepcopy(model)
    mask_model_params = encoding_models.ModelParams(domain_dim=2, output_channels=256, num_frequencies=256,
                                                    hidden_dim=512, std=5., num_layers=3)
    weight_tensor = torch.log(
        ((model.model.encode.frequencies**2).sum(0)**0.5))
    control_params_2 = encoding_controler.ControlParams(
        num_iterations=1000, epsilon=1e-5)

    if LEARN_MASK:
        mask_model = encoding_controler.get_controlled_model(
            mask_model_params, ENCODING_TYPE, control_params_2, ControllerType.NoControl).to(device)
        optMask = MaskModel(mask_model, model, weight_tensor,
                            lambda_cost=0.003, mask_lr=1e-4)
        mask = optMask.fit(vs_in, labels, target_image, out_path, tag, 3000,
                           vs_base=vs_base).detach()

        torch.save(mask, out_path / 'mask.pt')
        torch.save(mask_model.state_dict(), out_path / f'mask_model_{tag}.pt')
    else:
        mask = torch.load(out_path / 'mask.pt')
        mask_model = encoding_controler.get_controlled_model(
            mask_model_params, ENCODING_TYPE, control_params_2, ControllerType.NoControl).to(device)
        mask_model.load_state_dict(torch.load(
            out_path / f'mask_model_{tag}.pt'))
        optMask = MaskModel(mask_model, model, weight_tensor, lambda_cost=0.16)

    if RETRAIN:
        # only retrain last layer
        for param in model.parameters():
            param.requires_grad = False
        for param in model.model.model.model[-3:].parameters():
            param.requires_grad = True
        model2 = optimize(ENCODING_TYPE, model_params, CONTROLLER_TYPE, control_params, group, tag, out_path, device,
                          50, verbose=True, mask=mask, model=model, mask_model=optMask, lr=1e-4)
        torch.save(model2.state_dict(), out_path / f'model2_{tag}.pt')
    else:
        model2 = encoding_controler.get_controlled_model(
            model_params, ENCODING_TYPE, control_params, CONTROLLER_TYPE).to(device)
        model2.load_state_dict(torch.load(out_path / f'model2_{tag}.pt'))

    # Evaluation
    image_labels = image_labels.to(device)
    vs_in, labels = vs_in.to(device), labels.to(device)

    vs_base = vs_base.to(device)
    # masked_cords = masked_cords.to(device)
    # masked_labels = masked_labels.to(device)

    _, mask_base = optMask(vs_base)
    img_shape = target_image.shape
    # _, masked_mask = optMask(masked_cords)

    print(f'train psnr PRETRAIN {evaluate(model, vs_in, labels, img_shape)}')
    print(f'train psnr MASK {evaluate(model, vs_in, labels, img_shape, mask)}')
    print(f'train psnr RETRAIN {evaluate(model2, vs_in, labels, img_shape, mask)}')
    print()
    print(f'test psnr PRETRAIN {evaluate(model, vs_base, image_labels, img_shape)}')
    print(
        f'test psnr MASK {evaluate(model, vs_base, image_labels, img_shape, mask_base)}')
    print(
        f'test (psnr, ssim) RETRAIN {evaluate(model2, vs_base, image_labels, img_shape, mask_base)}')
    # print()
    # print('test masked psnr PRETRAIN', evaluate(model, masked_cords, masked_labels))
    # print('test masked psnr MASK', evaluate(model, masked_cords, masked_labels, masked_mask))
    # print('test masked psnr RETRAIN', evaluate(model2, masked_cords, masked_labels, masked_mask))

    return 0


if __name__ == '__main__':
    exit(main())
