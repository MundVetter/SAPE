from custom_types import *
from models import encoding_controller, encoding_models
from utils import files_utils, train_utils, image_utils
import constants
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import imageio
import os
import matplotlib.pyplot as plt
import sktime
from sktime.datasets import load_from_tsfile_to_dataframe
from pathlib import Path
import wandb

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
    vs_base, labels, vs_test, labels_test = vs_base.to(device), torch.from_numpy(labels).to(device), vs_test.to(device), torch.from_numpy(labels_test).to(device)
    lr = 1e-5
    model = encoding_controller.get_controlled_model(model_params, encoding_type, control_params, controller_type).to(device)
    tag = f'{encoding_type.value}_{controller_type.value}'
    if encoding_type is EncodingType.NoEnc:
        lr = 1e-4
    block_iterations = model.block_iterations
    out_path = f'{constants.CHECKPOINTS_ROOT}/1d/{name}/'
    os.makedirs(f'{out_path}', exist_ok=True)
    opt = Optimizer(model.parameters(), lr=lr)
    logger = train_utils.Logger().start(control_params.num_iterations, tag=tag)
    plot_character_trajectory(vs_base, labels[:, 0], labels[:, 1], labels[:, 2],  f"{tag}_train", out_path)
    plot_character_trajectory(vs_test[:, 0], labels_test[:, 0], labels_test[:, 1], labels_test[:, 2], f"{tag}_test", out_path)
    test_loss = 0

    for i in range(control_params.num_iterations):
        opt.zero_grad()
        out = model(vs_base)
        loss_all = nnf.mse_loss(out, labels, reduction='none')
        loss = loss_all.mean()
        loss.backward()
        opt.step()
        model.stash_iteration(loss_all.mean(-1))
        if block_iterations > 0 and (i + 1) % block_iterations == 0:
            model.update_progress()
        logger.reset_iter('loss', loss)
        logger.stash_iter('loss_test', test_loss)
        if verbose and ((i + 1) % freq == 0 or i == 0):
            with torch.no_grad():
                model.eval()
                out = model(vs_base)
                # aprox_func = torch.cat((vs_base, out), dim=1)
                plot_character_trajectory(vs_base, out[:, 0], out[:, 1], out[:, 2], f"{tag}_{i}_train", out_path)
                test_out = model(vs_test)
                # aprox_func = torch.cat((vs_base, out), dim=1)
                plot_character_trajectory(vs_test, test_out[:, 0], test_out[:, 1], test_out[:, 2], f"{tag}_{i}_test", out_path)
                if model.is_progressive:
                    _, mask_base = model(vs_base, get_mask=True)
                    if mask_base.dim() == 1:
                        mask_base = mask_base.unsqueeze(0).expand(vs_base.shape[0], mask_base.shape[0])
                    hm_base = mask_base.sum(1) / mask_base.shape[1]
                    hm_base = image_utils.to_heatmap(hm_base)
                    # export_poly(aprox_func, torch.cat((vs_target, labels_target), dim=1), f'{out_path}heatmap_{tag}/{i:05d}.png',
                                # colors=(None, hm_base))
                # calculate the loss on the test data
                test_loss = nnf.mse_loss(test_out, labels_test)
                logger.stash_iter('loss_test', test_loss)
                model.train()
    logger.stop()
    with torch.no_grad():
        final_test_loss = model(vs_test)
        final_test_loss = nnf.mse_loss(final_test_loss, labels_test)
        print(f'Final test loss: {final_test_loss}')

        # render the model with 2000 samples
        vs_base = torch.linspace(-1, 1, 2000).unsqueeze(-1).to(device)
        model.eval()
        out = model(vs_base)
        plot_character_trajectory(vs_base, out[:, 0], out[:, 1], out[:, 2], f"{tag}_final", out_path)



    files_utils.save_model(model, f'{out_path}model_{tag}.pth')
    if verbose and model.is_progressive:
        image_utils.gifed(f'{out_path}heatmap_{tag}/', .03, tag, reverse=False)
        files_utils.delete_all(f'{out_path}heatmap_{tag}/', '.png',
                               filter_out=lambda x: f'{control_params.num_iterations - 1}' == x[1])
    if verbose:
        image_utils.gifed(f'{out_path}opt_{tag}/', .03, tag, reverse=False)
        files_utils.delete_all(f'{out_path}opt_{tag}/', '.png',
                               filter_out=lambda x: f'{control_params.num_iterations - 1}' == x[1])

def plot_character_trajectory(time, x, y, downforce, i, out_path):
    fig = plt.figure(figsize=(10, 10))
    time, x, y, downforce = time.to('cpu').numpy(), x.to('cpu').numpy(), y.to('cpu').numpy(), downforce.to('cpu').numpy()
    x = np.cumsum(x)
    y = np.cumsum(y)
    downforce = np.cumsum(downforce)
    downforce_scaled = (downforce - downforce.min()) / downforce.max()
    plt.scatter(x, y, c = (time + 1) * 100, cmap='rainbow', s = downforce_scaled * 10)
    # plt.title(f'Character Trajectory of {label}')
    plt.xlabel('x')
    plt.ylabel('y') 
    plt.savefig(Path(out_path)  / f'{i}.png')
    plt.close(fig)


def main(PATH="signals/CharacterTrajectories_TEST.ts", INDEX = 4, DROP_MODE = False, CONTROLLER = ControllerType.SpatialProgressionStashed) -> int:
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
                "controller": CONTROLLER
            }
        )

    image_path = constants.DATA_ROOT / PATH
    os.makedirs(constants.CHECKPOINTS_ROOT, exist_ok=True)
    name = files_utils.split_path(PATH)[1]

    out_path = constants.CHECKPOINTS_ROOT / '1d' / name
    os.makedirs(out_path, exist_ok=True)

    # load with sktime
    data, cat = load_from_tsfile_to_dataframe(image_path)


    # create a interval between -1 and 1 for each sequence
    serie = torch.linspace(-1, 1, len(data["dim_0"][INDEX])).unsqueeze(-1)

    device = CUDA(0)
    encoding_type = EncodingType.FF
    controller_type = CONTROLLER


    num_samples = len(serie)
    control_params = encoding_controller.ControlParams(num_iterations=20_000, epsilon=1e-5, res=num_samples // 2)
    model_params = encoding_models.ModelParams(domain_dim=1, output_channels=3, num_freqs=256,
                                            hidden_dim=32, std=5., num_layers=2)
    labels = data["dim_0"][INDEX], data["dim_1"][INDEX], data["dim_2"][INDEX]
    # get into one array
    labels = np.array(labels).T.astype(np.float32)

    # use half of the data for training and half for testing
    vs_train = serie[::2]
    vs_test = serie[1::2]

    labels_train = labels[::2]
    labels_test = labels[1::2]
    # drop 70% of train randomly
    drop = np.random.choice(len(vs_train), int(len(vs_train) * 0.3), replace=False)
    vs_train = np.delete(vs_train, drop, axis=0)
    labels_train = np.delete(labels_train, drop, axis=0)

    optimize((vs_train, labels_train), (vs_test, labels_test), encoding_type, model_params, controller_type, control_params, num_samples, device, freq=500, verbose=True, name=f"{cat[INDEX]}_{INDEX}_{DROP_MODE}")
    return 0


if __name__ == '__main__':
    exit(main())