from utils.image_utils import init_source_target
from custom_types import *
from models import encoding_controler, encoding_models
from utils import files_utils, train_utils, image_utils
import constants
import copy
import matplotlib.pyplot as plt
import os

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
        self.encoding_dim  = frozen_model.encoding_dim
        self.weight_tensor = weight_tensor

    def fit(self, vs_in, labels, device, num_iterations=1000, vs_base=None):
        # Freeze the parameters of the frozen model
        for param in self.frozen_model.parameters():
            param.requires_grad = False

        logger = train_utils.Logger().start(num_iterations)
        vs_in, labels = vs_in.to(device), labels.to(device)
        for i in range(num_iterations):
            self.optimizer.zero_grad()

            mask_original, mask = self.forward(vs_in)
            frozen_model_output = self.frozen_model(vs_in, override_mask=mask)

            mse_loss = nnf.mse_loss(frozen_model_output, labels, reduction='none')
            mse_loss = mse_loss.mean()
            
            # Multiply the mask with the weight tensor before calculating the cost
            weighted_mask = mask_original * self.weight_tensor
            mask_cost = self.lambda_cost * weighted_mask.mean()
            total_loss = mse_loss + mask_cost
            logger.stash_iter('mse_train', mse_loss)

            if i % 100 == 0 and vs_base is not None:
                with torch.no_grad():
                    class shape:
                        def __init__(self):
                            self.shape = [512, 512, 3]
                    shape_obj = shape()
                    out, hm = plot_image(self.frozen_model, vs_base.to(device), shape_obj, self)
                    name = 'chibi'
                    tag = 'chibi_ff_no_control'
                    out_path = f'{constants.constants.CHECKPOINTS_ROOT}/2d_images/{name}/'
                    files_utils.export_image(out, f'{out_path}opt_{tag}/{i:04d}.png')
                    if hm is not None:
                        files_utils.export_image(hm, f'{out_path}heatmap_{tag}/{i:04d}.png')

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
        mask = torch.stack([mask_original, mask_original], dim=2).view(-1, self.encoding_dim - 2)
        ones = torch.ones_like(vs_in, device = vs_in.device)
        mask = torch.cat([ones, mask], dim=-1)
        return mask_original, mask


def optimize(image_path: Union[ARRAY, str], encoding_type: EncodingType, model_params,
             controller_type: ControllerType, control_params: encoding_controler.ControlParams, group, device: D,
             freq: int, verbose=False, mask=None, model=None, mask_model = None, lr=1e-3):
    name = files_utils.split_path(image_path)[1]
    vs_base, vs_in, labels, target_image, image_labels, masked_image = group
    tag = f'{name}_{encoding_type.value}_{controller_type.value}'
    out_path = f'{constants.CHECKPOINTS_ROOT}/2d_images/{name}/'

    if model is None:
        model = encoding_controler.get_controlled_model(model_params, encoding_type, control_params, controller_type).to(device)
    block_iterations = model.block_iterations
    vs_base, vs_in, labels, image_labels = vs_base.to(device), vs_in.to(device), labels.to(device), image_labels.to(device)
    opt = Optimizer(model.parameters(), lr=lr)
    logger = train_utils.Logger().start(control_params.num_iterations, tag=tag)
    files_utils.export_image(target_image, f'{out_path}target.png')
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
            with torch.no_grad():
                out, hm = plot_image(model, vs_base, target_image, mask_model)
                if hm is not None:
                    files_utils.export_image(hm, f'{out_path}heatmap_{tag}/{i:04d}.png')
                files_utils.export_image(out, f'{out_path}opt_{tag}/{i:04d}.png')
        logger.reset_iter()
    logger.stop()
    files_utils.save_model(model, f'{out_path}model_{tag}.pth')

    if verbose:
        image_utils.gifed(f'{out_path}opt_{tag}/', .07, tag, reverse=False)
        if model.is_progressive:
            image_utils.gifed(f'{out_path}heatmap_{tag}/', .07, tag, reverse=False)
            files_utils.delete_all(f'{out_path}heatmap_{tag}/', '.png',
                                   filter_out=lambda x: f'{control_params.num_iterations - 1}' == x[1])
        files_utils.delete_all(f'{out_path}opt_{tag}/', '.png',
                               filter_out=lambda x: f'{control_params.num_iterations - 1}' == x[1])
    return model

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def evaluate(model, vs_in, labels, mask = None):
    model.eval()
    with torch.no_grad():
        out = model(vs_in, override_mask=mask)
        return psnr(out, labels)


def main(KEEP_GROUP = False,
         PRETRAIN = True,
         LEARN_MASK = True,
         RETRAIN = True,
         IMAGE_PATH = "assets/images/chibi.jpg") -> int:
    device = CUDA(0)
    os.makedirs(constants.CHECKPOINTS_ROOT, exist_ok=True)
    print(device)
    name = files_utils.split_path(IMAGE_PATH)[1]
    scale = .25

    if KEEP_GROUP:
        group = torch.load(os.path.join(constants.CHECKPOINTS_ROOT, 'group.pt'))
    else:
        group = init_source_target(IMAGE_PATH, name, scale=scale, max_res=512, square=False, non_uniform_sampling=False)
        torch.save(group, os.path.join(constants.CHECKPOINTS_ROOT, 'group.pt'))
    vs_base, vs_in, labels, target_image, image_labels, (masked_cords, masked_labels, masked_image) = group

    model_params = encoding_models.ModelParams(domain_dim=2, output_channels=3, num_frequencies=256,
                                               hidden_dim=256, std=20., num_layers=3)
    control_params = encoding_controler.ControlParams(num_iterations=3000, epsilon=1e-3, res=128)
    encoding_type = EncodingType.FF
    controller_type = ControllerType.GlobalProgression

    if PRETRAIN:
        model = optimize(IMAGE_PATH, encoding_type, model_params, controller_type, control_params, group, device,
                    50, verbose=True)
        torch.save(model.state_dict(), os.path.join(constants.CHECKPOINTS_ROOT, 'model.pt'))
    else:
        model = encoding_controler.get_controlled_model(model_params, encoding_type, control_params, controller_type).to(device)
        model.load_state_dict(torch.load(os.path.join(constants.CHECKPOINTS_ROOT, 'model.pt')))

    # model_copy = copy.deepcopy(model)
    mask_model_params = encoding_models.ModelParams(domain_dim=2, output_channels=256, num_frequencies=256,
                                                hidden_dim=512, std=5., num_layers=3)
    weight_tensor = torch.log(((model.model.encode.frequencies**2).sum(0)**0.5))
    control_params_2 = encoding_controler.ControlParams(num_iterations=1000, epsilon=1e-5)

    if LEARN_MASK:
        mask_model = encoding_controler.get_controlled_model(mask_model_params, encoding_type, control_params_2, ControllerType.NoControl).to(device)
        optMask = MaskModel(mask_model, model, weight_tensor, lambda_cost=0.003, mask_lr=1e-4)
        mask = optMask.fit(vs_in, labels, device, 3000, vs_base=vs_base).detach()

        torch.save(mask, os.path.join(constants.CHECKPOINTS_ROOT, 'mask.pt'))
        torch.save(mask_model.state_dict(), os.path.join(constants.CHECKPOINTS_ROOT, 'mask_model.pt'))
    else:
        mask = torch.load(os.path.join(constants.CHECKPOINTS_ROOT, 'mask.pt'))
        mask_model = encoding_controler.get_controlled_model(mask_model_params, encoding_type, control_params_2, ControllerType.NoControl).to(device)
        mask_model.load_state_dict(torch.load(os.path.join(constants.CHECKPOINTS_ROOT, 'mask_model.pt')))
        optMask = MaskModel(mask_model, model, weight_tensor, lambda_cost=0.16)

    if RETRAIN:
        model2 = optimize(IMAGE_PATH, encoding_type, model_params, controller_type, control_params, group, device,
                    50, verbose=True, mask=mask, model=model, mask_model = optMask, lr=1e-4)
        torch.save(model2.state_dict(), os.path.join(constants.CHECKPOINTS_ROOT, 'model2.pt'))
    else:
        model2 = encoding_controler.get_controlled_model(model_params, encoding_type, control_params,controller_type).to(device)
        model2.load_state_dict(torch.load(os.path.join(constants.CHECKPOINTS_ROOT, 'model2.pt')))

    ### Evaluation
    image_labels = image_labels.to(device)
    vs_in, labels = vs_in.to(device),labels.to(device)
    
    vs_base = vs_base.to(device)
    # masked_cords = masked_cords.to(device)
    # masked_labels = masked_labels.to(device)

    _, mask_base = optMask(vs_base)
    # _, masked_mask = optMask(masked_cords)

    print(f'train psnr PRETRAIN {evaluate(model, vs_in, labels)}')
    print(f'train psnr MASK {evaluate(model, vs_in, labels, mask)}')
    print(f'train psnr RETRAIN {evaluate(model2, vs_in, labels, mask)}')
    print()
    print(f'test psnr PRETRAIN {evaluate(model, vs_base, image_labels)}')
    print(f'test psnr MASK {evaluate(model, vs_base, image_labels, mask_base)}')
    print(f'test psnr RETRAIN {evaluate(model2, vs_base, image_labels, mask_base)}')
    # print()
    # print('test masked psnr PRETRAIN', evaluate(model, masked_cords, masked_labels))
    # print('test masked psnr MASK', evaluate(model, masked_cords, masked_labels, masked_mask))
    # print('test masked psnr RETRAIN', evaluate(model2, masked_cords, masked_labels, masked_mask))

    return 0

if __name__ == '__main__':
    exit(main())
