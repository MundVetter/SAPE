import os
from models.encoding_models import MaskModel, evaluate
from utils.files_utils import save_results_to_csv
from utils.image_utils import log_evaluation_progress, psnr, ssim
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'

from utils.image_utils import init_source_target
from custom_types import *
from models import encoding_controller, encoding_models
from utils import files_utils, train_utils
import constants
import copy
import matplotlib.pyplot as plt
import os
from pathlib import Path
import wandb


def optimize(encoding_type: EncodingType, model_params,
             controller_type: ControllerType, control_params: encoding_controller.ControlParams, group, tag, out_path, device: D,
             freq: int, verbose=False, mask=None, model=None, mask_model=None, lr=1e-3, eval_labels = None, compensate_inv_prob = False):
    # TODO: change to original SAPE
    vs_base, vs_in, labels, target_image, image_labels, _, prob = group
    model_provided = True
    if model is None:
        model = encoding_controller.get_controlled_model(
            model_params, encoding_type, control_params, controller_type).to(device)
        model_provided = False
    wandb.watch(model)
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
        if not model_provided:
            model.stash_iteration(loss_all.mean(-1))
        
        if compensate_inv_prob:
            loss_all[:, 0] *= inv_prob
            loss_all[:, 1] *= inv_prob
            loss_all[:, 2] *= inv_prob
        loss = loss_all.mean()
        if i == 0:
            print(loss)
        loss.backward()
        opt.step()
        logger.stash_iter('mse_train', loss)
        wandb.log({'mse_train': loss})

        if not model_provided and block_iterations > 0 and (i + 1) % block_iterations == 0:
            model.update_progress()
        if (((i + 1) % freq == 0) or (i == 0)) and verbose:
            log_evaluation_progress(model, target_image, out_path, tag, vs_base, device, i = i, labels = eval_labels)
        logger.reset_iter()
    logger.stop()

    return model

def main(NON_UNIFORM=True,
         EPOCHS=1,
         PATH="images/chibi.jpg",
         ENCODING_TYPE = EncodingType.FF,
         CONTROLLER_TYPE = ControllerType.SpatialProgressionStashed,
         MASK_RES = 512,
         LAMBDA_COST = 0.1,
         WEIGHT_DECAY = 1,
         RUN_NAME=None,
         LR = 1e-3,
         THRESHOLD = 1,
         SIGMA = 20.,
         INV_PROB = True, **kwargs) -> int:

    if constants.DEBUG:
        wandb.init(mode="disabled")
    else:
        wandb.init(project="2d_image",
                   group=RUN_NAME,
            config={
                "non_uniform": NON_UNIFORM,
                "image_path": PATH,
                "encoding_type": ENCODING_TYPE,
                "controller_type": CONTROLLER_TYPE,
                "mask res": MASK_RES,
                "threshold": THRESHOLD,
                "lr": LR,
                "epochs": EPOCHS,
                "sigma": SIGMA,
                "inv prob": INV_PROB
            })
        wandb.run.log_code(".")

    device = CUDA(0)
    print(device)

    image_path = constants.DATA_ROOT / PATH
    os.makedirs(constants.CHECKPOINTS_ROOT, exist_ok=True)
    name = files_utils.split_path(PATH)[1]

    scale = .25
    group = init_source_target(image_path, name, scale=scale,
                               max_res=512, square=False, non_uniform_sampling=NON_UNIFORM)
    vs_base, vs_in, labels, target_image, image_labels, (masked_cords, masked_labels, masked_image), prob = group

    # NEEDS TO BE CHANGED
    model_params = encoding_models.ModelParams(domain_dim=2, output_channels=3, num_frequencies=256,
                                               hidden_dim=256, std=SIGMA, num_layers=2, use_id_encoding=True)

    tag_without_filename = f"{ENCODING_TYPE.value}_{MASK_RES}_{CONTROLLER_TYPE.value}_{NON_UNIFORM}_{RUN_NAME}"
    tag = f"{name}_{tag_without_filename}"

    out_path = constants.CHECKPOINTS_ROOT / '2d_images' / name
    os.makedirs(out_path, exist_ok=True)

    if CONTROLLER_TYPE == ControllerType.LearnableMask:
        mask_model_params = copy.deepcopy(model_params)
        mask_model_params.output_channels = 256
        mask_model_params.std = SIGMA / 4

        cmlp = encoding_controller.get_controlled_model(
            model_params, ENCODING_TYPE, encoding_controller.ControlParams(), ControllerType.NoControl).to(device)
        
    
        mask_model = encoding_controller.get_controlled_model(
            mask_model_params, ENCODING_TYPE, encoding_controller.ControlParams(), ControllerType.NoControl).to(device)

        model = MaskModel(mask_model, cmlp, prob,
                            lambda_cost=LAMBDA_COST, mask_act=torch.erf, threshold = THRESHOLD, compensate_inv_prob=INV_PROB)
        mask = model.fit(vs_in, labels, target_image, out_path, tag, EPOCHS,
                           vs_base=vs_base, lr=LR, weight_decay = WEIGHT_DECAY, eval_labels=image_labels, log = log_evaluation_progress).detach()

        torch.save(mask, out_path / f'mask_{tag}.pt')
    else:
        control_params = encoding_controller.ControlParams(
        num_iterations=EPOCHS, epsilon=LR, res=MASK_RES)
        model = optimize(ENCODING_TYPE, model_params, CONTROLLER_TYPE, control_params, group, tag, out_path, device,
                         100, verbose=True, eval_labels = image_labels, compensate_inv_prob = INV_PROB)

    torch.save(model.state_dict(), out_path / f'model_{tag}.pt')

    res_train = evaluate(model, vs_in.to(device), labels.to(device), psnr)
    res_test = evaluate(model, vs_base.to(device), image_labels.to(device), psnr)
    res_test_ssim = evaluate(model, vs_base.to(device), image_labels.to(device), ssim, img_shape = [512, 512])
    res_masked = evaluate(model, masked_cords.to(device), masked_labels.to(device), psnr)

    print(f"TRAIN PSNR: {res_train}")
    print(f"TEST PSNR: {res_test}")
    print(f"TEST SSIM: {res_test_ssim}")
    print(f"TEST MASKED PSNR: {res_masked}")

    wandb.log({"train_psnr": res_train, "test_psnr": res_test, "test_ssim": res_test_ssim, "test_masked_psnr": res_masked})

    save_results_to_csv([
        ("train", res_train), 
        ("test", res_test), 
        ("test_ssim", res_test_ssim), 
        ("test_masked", res_masked)
    ], out_path, tag_without_filename)

    return 0


if __name__ == '__main__':
    exit(main())
