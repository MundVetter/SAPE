import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'

from custom_types import *
from models import encoding_controller, encoding_models
from utils import files_utils, train_utils, image_utils
import constants
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import imageio

import matplotlib.pyplot as plt
import sktime
from sktime.datasets import load_from_tsfile_to_dataframe
from pathlib import Path
from utils.image_utils import psnr
import wandb
import copy


plt.ioff()

class Function:

    def __call__(self, x: T) -> T:
        return self.function(x)

    @property
    def name(self):
        return self.function.__name__

    def __init__(self, function: Callable[[T], T]):
        self.function = function
        self.samples = None

def optimize(train, test, encoding_type: EncodingType, model_params,
             controller_type: ControllerType, control_params,
             num_samples: int, device: D,
             freq=500, verbose=False, name='default'):

    (vs_base, labels) = train
    (vs_test, labels_test) = test
    vs_base, labels, vs_test, labels_test = torch.from_numpy(vs_base).to(device), torch.from_numpy(labels).to(device), torch.from_numpy(vs_test).to(device), torch.from_numpy(labels_test).to(device)
    lr = 1e-4

    if controller_type is ControllerType.LearnableMask:
        model = encoding_controller.get_controlled_model(model_params, encoding_type, control_params, ControllerType.NoControl).to(device)
        mask_model_params = copy.deepcopy(model_params)
        mask_model_params.output_channels = 256
        mask_model_params.std = 0.5
        mask_model = encoding_controller.get_controlled_model(mask_model_params, encoding_type, control_params, ControllerType.NoControl).to(device)
        model = encoding_models.MaskModel(mask_model, model).to(device)
    else:
        model = encoding_controller.get_controlled_model(model_params, encoding_type, control_params, controller_type).to(device)
        block_iterations = model.block_iterations

    tag = f'{encoding_type.value}_{controller_type.value}'
    if encoding_type is EncodingType.NoEnc:
        lr = 1e-3
   
    out_path = f'{constants.CHECKPOINTS_ROOT}/1d/{name}/'
    os.makedirs(f'{out_path}', exist_ok=True)
    if ControllerType.LearnableMask == controller_type:
        opt = OptimizerW(model.parameters(), lr=lr, weight_decay=1)
    else:
        opt = Optimizer(model.parameters(), lr=lr)
    logger = train_utils.Logger().start(control_params.num_iterations, tag=tag)
    # plot_character_trajectory(vs_base, labels[:, 0], labels[:, 1], labels[:, 2],  f"{tag}", "train_input", out_path)
    # plot_character_trajectory(vs_test[:, 0], labels_test[:, 0], labels_test[:, 1], labels_test[:, 2], f"{tag}", "test_input", out_path)
    test_loss = 0

    for i in range(control_params.num_iterations):
        opt.zero_grad()
        if ControllerType.LearnableMask == controller_type:
            model.train_iter(vs_base, labels, logger)
            opt.step()
        else:
            out = model(vs_base)
            loss_all = nnf.mse_loss(out, labels, reduction='none')
            loss = loss_all.mean()
            loss.backward()
            opt.step()
            model.stash_iteration(loss_all.mean(-1))
            wandb.log({'mse_train': loss.mean()})
            logger.reset_iter('mse_train', loss.mean())
    
            if block_iterations > 0 and (i + 1) % block_iterations == 0:
                model.update_progress()

        logger.stash_iter('PSNR test', test_loss)
        if verbose and ((i + 1) % freq == 0 or i == 0):
            with torch.no_grad():
                model.eval()
                # aprox_func = torch.cat((vs_base, out), dim=1)
                # plot_character_trajectory(vs_base[:, 0], out[:, 0], out[:, 1], out[:, 2], f"{tag}_{i}", "train", out_path)
                test_out = model(vs_test)
                # aprox_func = torch.cat((vs_base, out), dim=1)
                # plot_character_trajectory(vs_test[:, 0], test_out[:, 0], test_out[:, 1], test_out[:, 2], f"{tag}_{i}", "test", out_path)
                # if model.is_progressive:
                #     _, mask_base = model(vs_base, get_mask=True)
                #     if mask_base.dim() == 1:
                #         mask_base = mask_base.unsqueeze(0).expand(vs_base.shape[0], mask_base.shape[0])
                #     hm_base = mask_base.sum(1) / mask_base.shape[1]
                #     hm_base = image_utils.to_heatmap(hm_base)
                    # export_poly(aprox_func, torch.cat((vs_target, labels_target), dim=1), f'{out_path}heatmap_{tag}/{i:05d}.png',
                                # colors=(None, hm_base))
                # calculate the loss on the test data
                test_loss = psnr(test_out, labels_test)
                logger.stash_iter('PSNR test', test_loss)
                wandb.log({'PSNR test': test_loss})

                for i in torch.linspace(-1, 1, 10):
                    # render the model with 2000 samples
                    base_base = torch.linspace(-1, 1, 2000).unsqueeze(-1).to(device)
                    # add test index to the base
                    base_in = torch.cat((base_base, torch.ones_like(base_base) * i), dim=1)
                    model.eval()
                    out = model(base_in)
                    if i == -1:
                        plot_character_trajectory(vs_base, out[:, 0], out[:, 1], out[:, 2], f"{tag}", f"intermediate_{i}", out_path, labels_test[:72])
                    else:
                        plot_character_trajectory(vs_base, out[:, 0], out[:, 1], out[:, 2], f"{tag}", f"intermediate_{i}", out_path)
                model.train()
    logger.stop()
    with torch.no_grad():
        out = model(vs_test)
        final_test_loss = psnr(out, labels_test) 
        print(f'PSNR TEST FINAL: {final_test_loss}')
        wandb.log({'PSNR TEST FINAL': final_test_loss})


        for i in torch.linspace(-1, 1, 10):
            # render the model with 2000 samples
            vs_base = torch.linspace(-1, 1, 2000).unsqueeze(-1).to(device)
            # add test index to the base
            base_in = torch.cat((vs_base, torch.ones_like(vs_base) * i), dim=1)
            model.eval()
            out = model(base_in)
            if i == -1:
                plot_character_trajectory(vs_base, out[:, 0], out[:, 1], out[:, 2], f"{tag}", f"intermediate_{i}", out_path, labels_test[:72])
            else:
                plot_character_trajectory(vs_base, out[:, 0], out[:, 1], out[:, 2], f"{tag}", f"intermediate_{i}", out_path)



    files_utils.save_model(model, f'{out_path}model_{tag}.pth')
    if verbose and model.is_progressive:
        image_utils.gifed(f'{out_path}heatmap_{tag}/', .03, tag, reverse=False)
        files_utils.delete_all(f'{out_path}heatmap_{tag}/', '.png',
                               filter_out=lambda x: f'{control_params.num_iterations - 1}' == x[1])
    if verbose:
        image_utils.gifed(f'{out_path}opt_{tag}/', .03, tag, reverse=False)
        files_utils.delete_all(f'{out_path}opt_{tag}/', '.png',
                               filter_out=lambda x: f'{control_params.num_iterations - 1}' == x[1])

def plot_character_trajectory(time, x, y, downforce, i, tag, out_path, overlay_point = None):
    fig = plt.figure(figsize=(10, 10))
    time, x, y, downforce = time.to('cpu').numpy(), x.to('cpu').numpy(), y.to('cpu').numpy(), downforce.to('cpu').numpy()
    # x = np.cumsum(x)
    # y = np.cumsum(y)
    # downforce = np.cumsum(downforce)
    # downforce_scaled = downforce
    plt.scatter(x, y, c = (time + 1) * 100, cmap='rainbow', s = downforce * 10)
    if overlay_point is not None:
        overlay_point = overlay_point.to('cpu').numpy()
        plt.scatter(overlay_point[:, 0], overlay_point[:, 1], c = 'black', s = 20)
    # hide axis
    plt.axis('off')
    # plt.title(f'Character Trajectory of {label}')
    path = Path(out_path)  / f'{i}.png'
    plt.savefig(path)
    plt.close(fig)
    wandb.log({f'character_trajectory {tag}': wandb.Image(str(path))})
    


def main(PATH="signals/CharacterTrajectories_TEST.ts", INDEX = 100, DROP_MODE = True, CONTROLLER = ControllerType.LearnableMask, ENCODING = EncodingType.FF) -> int:
    if constants.DEBUG:
        wandb.init(mode="disabled")
    else:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="1d",
            # set the config parameters
            config={
                "path": PATH,
                "index": INDEX,
                "drop_mode": DROP_MODE,
                "controller": CONTROLLER,
                "encoding": ENCODING
            }
        )

    image_path = constants.DATA_ROOT / PATH
    os.makedirs(constants.CHECKPOINTS_ROOT, exist_ok=True)
    name = files_utils.split_path(PATH)[1]

    out_path = constants.CHECKPOINTS_ROOT / '1d' / name
    os.makedirs(out_path, exist_ok=True)

    # load with sktime
    data, cat = load_from_tsfile_to_dataframe(image_path)

    if True:
        series = []
        labels_agg = []

        for INDEX in range(len(data["dim_0"])):
            interval_val = (INDEX / len(data["dim_0"])) * 2 - 1

            # create a interval between -1 and 1 for each sequence
            serie = torch.linspace(-1, 1, len(data["dim_0"][INDEX])).unsqueeze(-1)

            device = CUDA(0)
            encoding_type = ENCODING
            controller_type = CONTROLLER


            num_samples = len(serie)
            labels = data["dim_0"][INDEX], data["dim_1"][INDEX], data["dim_2"][INDEX]
            # get into one array
            labels = np.array(labels).T.astype(np.float32)

            # intrgrate the data
            labels = np.cumsum(labels, axis=0)
            # normalize the data for each dimension separately
            labels[:, 0] = (labels[:, 0] - labels[:, 0].min())
            labels[:, 0] /= labels[:, 0].max()
            labels[:, 1] = (labels[:, 1] - labels[:, 1].min())
            labels[:, 1] /= labels[:, 1].max()
            labels[:, 2] = (labels[:, 2] - labels[:, 2].min())
            labels[:, 2] /= labels[:, 2].max()

            for i in range(len(labels)):
                series.append(np.array([serie[i], interval_val]))
                labels_agg.append(labels[i])


        series = np.stack(series).astype(np.float32)
        labels_agg = np.array(labels_agg).astype(np.float32)
        # save the data
        np.save(f'{out_path}series.npy', series)
        np.save(f'{out_path}labels.npy', labels_agg)
    else:
        series = np.load(f'{out_path}series.npy')
        labels_agg = np.load(f'{out_path}labels.npy')

    # use half of the data for training and half for testing
    vs_train = series[::2]
    vs_test = series[1::2]

    labels_train = labels_agg[::2]
    labels_test = labels_agg[1::2]
    if DROP_MODE:
        # drop 70% of train randomly
        rng = np.random.default_rng(42)
        drop = rng.choice(len(vs_train), int(len(vs_train) * 0.7), replace=False)
        vs_train = np.delete(vs_train, drop, axis=0)
        labels_train = np.delete(labels_train, drop, axis=0)

    control_params = encoding_controller.ControlParams(num_iterations=4000, epsilon=1e-11, res=len(vs_train) // 2)
    if ControllerType.LearnableMask == controller_type:
        num_layers = 2
    else:
        num_layers = 4

    model_params = encoding_models.ModelParams(domain_dim=2, output_channels=3, num_freqs=256,
                                            hidden_dim=32, std=1.0, num_layers=num_layers)

    optimize((vs_train, labels_train), (vs_test, labels_test), encoding_type, model_params, controller_type, control_params, num_samples, device, freq=1000, verbose=True, name=f"{cat[INDEX]}_{INDEX}_{DROP_MODE}")
    return 0


if __name__ == '__main__':
    exit(main())