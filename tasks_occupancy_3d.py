from custom_types import *
import constants
from utils import mesh_utils, files_utils, train_utils, sdf_mesh, image_utils
from models import encoding_controller, encoding_models
import igl
import wandb
import copy
from pathlib import Path

def get_in_out(mesh: T_Mesh, points: T):
    vs, faces = mesh[0].numpy(), mesh[1].numpy()
    points = points.numpy()
    w = igl.winding_number(vs, faces, points)
    w = torch.from_numpy(w).float()
    labels = w.lt(.9).float()
    return labels


class MeshSampler(Dataset):

    def shuffle(self):
        order = torch.rand(self.points.shape[0]).argsort()
        self.points, self.labels = self.points[order], self.labels[order]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        return self.points[item], self.labels[item]

    def get_random_points(self, num_points: float) -> TS:
        random_points = (torch.rand(int(num_points), 3) * 2 - 1)
        labels = get_in_out(self.mesh, random_points).unsqueeze(1)
        return random_points, labels

    def get_surface_points(self, num_points: float, sigma: float, on_surface_points: Optional[T] = None) -> TS:
        if on_surface_points is None:
            on_surface_points = mesh_utils.sample_on_mesh(self.mesh, int(num_points), sample_s=mesh_utils.SampleBy.HYB)[0]
        near_points = on_surface_points + torch.randn(on_surface_points.shape) * sigma
        near_points = near_points.clamp_(-1, 1)
        labels = get_in_out(self.mesh, near_points).unsqueeze(1)
        return near_points, labels, on_surface_points

    def init_samples(self) -> TS:
        total = self.n_samples
        split = [float(part) / sum(self.split) for part in self.split]
        near_points_a, labels_a, on_surface_points = self.get_surface_points(int(total * split[0]), .01)
        near_points_b, labels_b, _ = self.get_surface_points(int(total * split[0]), .1, on_surface_points)
        random_points, labels_c, = self.get_random_points(int(total - near_points_a.shape[0] * 2))
        all_points = torch.cat((near_points_a, near_points_b, random_points), dim=0)
        labels = torch.cat((labels_a, labels_b, labels_c), dim=0)
        return all_points.cpu(), labels.cpu()

    def reset(self):
        if self.pointer >= len(self.data):
            self.data.append(self.init_samples())
            self.save_data()
        self.points, self.labels = self.data[self.pointer]
        self.points, self.labels = self.points.to(self.device), self.labels.to(self.device)
        self.shuffle()
        self.pointer = (self.pointer + 1) % self.buffer_size

    @staticmethod
    def load_data(mesh_path: str):
        name = files_utils.split_path(mesh_path)[1]
        cache_path = f"{constants.DATA_ROOT}/cache/sdf_{name}"
        data = files_utils.load_pickle(cache_path)
        return data

    def save_data(self):
        if self.buffer_size == len(self.data) and not self.cache_saved:
            name = files_utils.split_path(self.mesh_path)[1]
            files_utils.save_pickle(self.data, f"{constants.DATA_ROOT}/cache/sdf_{name}")
            self.cache_saved = True

    def load_mesh(self):
        mesh = files_utils.load_mesh(self.mesh_path)
        mesh = mesh_utils.to_unit_sphere(mesh, scale=.95)
        mesh = mesh_utils.triangulate_mesh(mesh)[0]
        return mesh

    def __init__(self, path, device: D, buffer_size=10, n_samples = 1e6):
        self.device = device
        self.name = files_utils.split_path(path)[1]
        self.data = None #self.load_data(path)
        self.cache_saved = False
        self.split = (3, 3, 3)
        self.buffer_size = buffer_size
        self.mesh_path = path
        self.pointer = 0
        self.mesh = self.load_mesh()
        if self.data is None:
            self.data = []
        else:
            self.cache_saved = len(self.data) >= self.buffer_size
        self.points: TN = None
        self.labels: TN = None
        self.n_samples = n_samples


def model_for_export(model) -> Callable[[T], T]:

    def call(x: T) -> T:
        out_: T = model(x)
        out_.sigmoid_()
        out_ = out_ - .5
        return out_
    model.eval()

    return call


def optimize(ds: MeshSampler, encoding_type: EncodingType = None, model_params: encoding_models.ModelParams = None,
             controller_type: ControllerType = None, control_params: encoding_controller.ControlParams = None,
             device: D = CPU, freq: int = 25, verbose=False, model = None, Opt = Optimizer, weight_decay = 0, custom_train = False, tag = '', out_path = 'checkpoints/3d_occupancy/', epochs = 1, batch_size = 5000, render_res = 256):


    name = ds.name

    ds.reset()
    in_iters = max(len(ds) // batch_size, 1)
    if model is None:
        model = encoding_controller.get_controlled_model(model_params, encoding_type, control_params, controller_type).to(device)
    lr = 1e-4
    opt = Opt(model.parameters(), lr=lr, weight_decay=weight_decay)
    logger = train_utils.Logger().start(epochs, tag=f"{name} {tag}")
    wandb.watch(model)
    for i in range(epochs):
        loss_train = 0
        for j in range(in_iters):
            if len(ds) <= batch_size:
                vs, labels = ds.points, ds.labels
            else:
                vs, labels = ds.points[j * batch_size: (j + 1) * batch_size], ds.labels[j * batch_size: (j + 1) * batch_size]
            opt.zero_grad()
            if custom_train:
                model.train_iter(vs, labels, logger)
                if i % freq == 0:
                    mask_size = encoding_models.mean_abs_weights(model.mask)
                    main_size = encoding_models.mean_abs_weights(model.cmlp)
                    wandb.log({'mask_size': mask_size, 'main_size': main_size})
            else:
                out = model(vs)
                loss_all = nnf.binary_cross_entropy_with_logits(out, labels, reduction='none')
                loss = loss_all.mean()
                loss.backward()
            opt.step()
            if not custom_train:
                model.stash_iteration(loss_all.mean(-1))
                loss_train += loss.item()
        if not custom_train:
            loss_train = float(loss_train) / in_iters
            wandb.log({'mse_train': loss_train})
            logger.reset_iter('mse_train', loss_train)
            model.update_progress()
        if len(ds) > batch_size:
            ds.reset()
        if (i + 1) % freq == 0 and verbose:
            print(i)
            sdf_mesh.create_mesh(model_for_export(model), f'{out_path}{tag}_meshes/{i:04d}', res=128, device=device)
            model.train()
    logger.stop()
    # model.load_state_dict(torch.load(f'{out_path}model_{tag}.pth', map_location=device))
    sdf_mesh.create_mesh(model_for_export(model), f'{out_path}final_{tag}', res=render_res, device=device)
    files_utils.save_model(model, f'{out_path}model_{tag}.pth')
    return model
    # if model.is_progressive:
    #     export_heatmap()

def intersect(vol1, vol2):
    return (vol1 & vol2).float().sum()

def union(vol1, vol2):
    return (vol1 | vol2).float().sum()

def evaluate(model, ds, batch_size = 5000):
    model.eval()
    ds.reset()
    in_iters = len(ds) // batch_size
    intersection = 0
    _union = 0
    with torch.no_grad():
        for j in range(in_iters):
            vs, labels = ds.points[j * batch_size: (j + 1) * batch_size], ds.labels[j * batch_size: (j + 1) * batch_size]
            out = model(vs)
            out = out.ge(0).int()
            labels = labels.int()
            intersection += intersect(out, labels)
            _union += union(out, labels)
    return intersection / _union

def main(EPOCHS=10,
         PATH="meshes/MalteseFalconSolid.stl",
         ENCODING_TYPE = EncodingType.FF,
         CONTROLLER_TYPE = ControllerType.LearnableMask,
         MASK_RES = 64,
         LAMBDA_COST = 0.1,
         WEIGHT_DECAY = 1,
         SIGMA = 20.,
         RUN_NAME=None,
         LR = 1e-4,
         THRESHOLD = 1,
         BATCH_SIZE = 5000,
         RENDER_RES = 256,
         BN = False,
         ID = False,
         MASK_SIGMA = 1,
         LAYERS = 4,
         N_SAMPLES = 1e6, **kwargs) -> int:

    if constants.DEBUG:
        wandb.init(mode="disabled")
    else:
        wandb.init(project="3d_occupancy",
                   group=RUN_NAME,
            config={
                "path": PATH,
                "encoding_type": ENCODING_TYPE,
                "controller_type": CONTROLLER_TYPE,
                "mask res": MASK_RES,
                "lr": LR,
                "epochs": EPOCHS,
                "batch size": BATCH_SIZE,
                "sigma": SIGMA,
                "bn": BN,
                "use_id": ID,
                "mask_sigma": MASK_SIGMA,
                "n samples": N_SAMPLES
            })
        wandb.run.log_code(".")

    device = CUDA(0)
    print(device)

    mesh_path = str(constants.DATA_ROOT / PATH)
    ds = MeshSampler(mesh_path, device, n_samples=N_SAMPLES)

    name = ds.name
    tag = f'{ENCODING_TYPE}_{CONTROLLER_TYPE}_{MASK_RES}_{RUN_NAME}'
    out_path = f'{constants.CHECKPOINTS_ROOT}/3d_occupancy/{name}/'

    model_params = encoding_models.ModelParams(domain_dim=3, output_channels=1, std=5., hidden_dim=256,
                                                num_layers=LAYERS, num_frequencies=256, use_id_encoding=ID, bn = BN)
    if CONTROLLER_TYPE == ControllerType.LearnableMask:
        mask_model_params = copy.deepcopy(model_params)
        mask_model_params.output_channels = 256
        mask_model_params.std = MASK_SIGMA

        cmlp = encoding_controller.get_controlled_model(
            model_params, ENCODING_TYPE, encoding_controller.ControlParams(), ControllerType.NoControl).to(device)
        mask_model = encoding_controller.get_controlled_model(
            mask_model_params, ENCODING_TYPE, encoding_controller.ControlParams(), ControllerType.NoControl).to(device)

        model = encoding_models.MaskModel(mask_model, cmlp, lambda_cost=LAMBDA_COST, mask_act=torch.erf, threshold = THRESHOLD, loss= nnf.binary_cross_entropy_with_logits, bn = BN)
        model = optimize(ds, device = device, freq = 250, verbose=True, tag = tag, out_path = out_path, model = model, Opt = OptimizerW, weight_decay = WEIGHT_DECAY, custom_train = True, epochs = EPOCHS, batch_size = BATCH_SIZE, render_res=RENDER_RES)
    else:
        control_params = encoding_controller.ControlParams(num_iterations=500, epsilon=1e-2, res=MASK_RES)
        model = optimize(ds, encoding_type=ENCODING_TYPE, model_params=model_params, controller_type=CONTROLLER_TYPE, control_params=control_params, device=device, freq=250, verbose=True, tag=tag, out_path=out_path, epochs=EPOCHS, batch_size=BATCH_SIZE, render_res=RENDER_RES)

    ds_eval = MeshSampler(mesh_path, device)
    result = evaluate(model, ds_eval, batch_size=BATCH_SIZE)

    print(f"{tag} IOU: ", result)
    wandb.log({'iou': result})

    files_utils.save_results_to_csv([
        ('3d_iou', result),
    ], Path(out_path), tag)


if __name__ == '__main__':
    exit(main())