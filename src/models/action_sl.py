import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import floor
from torch.utils.tensorboard import SummaryWriter
from yaml import safe_load
import os
from data_processing.data_utils import ImbalancedDatasetSampler
from data_processing.data_loaders import HDF5TorchDataset, load_data_iter
import matplotlib.pyplot as plt
from features.feat_utils import image_transforms

np.random.seed(42)
import gym
from gym.wrappers import FrameStack  # ,Monitor


class ACTION_SL(nn.Module):
    def __init__(
        self,
        input_shape=(84, 84),
        load_model=False,
        epoch=0,
        num_actions=18,
        game="breakout",
        data_types=["images", "actions", "gazes_fused_noop"],
        dataset_train="combined",
        dataset_train_load_type="disk",
        dataset_val="combined",
        dataset_val_load_type="disk",
        device=torch.device("cpu"),
        gaze_pred_model=None,
        mode="train",
    ):
        super(ACTION_SL, self).__init__()
        self.game = game
        self.data_types = data_types
        self.input_shape = input_shape
        self.num_actions = num_actions
        with open("src/config.yaml", "r") as f:
            self.config_yml = safe_load(f.read())

        model_save_dir = os.path.join(
            self.config_yml["MODEL_SAVE_DIR"], game, dataset_train
        )
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        self.model_save_string = os.path.join(
            model_save_dir, self.__class__.__name__ + "_Epoch_{}.pt"
        )
        log_dir = os.path.join(
            self.config_yml["RUNS_DIR"],
            game,
            "{}_{}".format(dataset_train, self.__class__.__name__),
        )
        self.writer = SummaryWriter(
            log_dir=os.path.join(
                log_dir,
                "run_{}".format(
                    len(os.listdir(log_dir)) if os.path.exists(log_dir) else 0
                ),
            )
        )
        self.device = device
        self.batch_size = self.config_yml["BATCH_SIZE"]
        self.gaze_gate = nn.GRU(3136, 1, 1, batch_first=True)

        # self.gate_hidden = (torch.ones(1, self.batch_size, 1) * -1).to(device=self.device)
        # self.gate_hidden.requires_grad= False
        # self.W = torch.nn.Parameter(torch.Tensor([1]),requires_grad=True)

        if mode != "eval":
            self.train_data_iter = load_data_iter(
                game=self.game,
                data_types=self.data_types,
                dataset=dataset_train,
                device=device,
                batch_size=self.batch_size,
                sampler=ImbalancedDatasetSampler,
                load_type=dataset_train_load_type,
                dataset_exclude=dataset_val,
            )

            self.val_data_iter = load_data_iter(
                game=self.game,
                data_types=self.data_types,
                dataset=dataset_val,
                dataset_exclude=dataset_train,
                device=device,
                batch_size=self.batch_size,
                load_type=dataset_val_load_type,
            )

        self.conv1 = nn.Conv2d(4, 32, 8, stride=(4, 4))
        self.pool = nn.MaxPool2d((1, 1), (1, 1), (0, 0), (1, 1))
        # self.pool = lambda x: x

        self.conv2 = nn.Conv2d(32, 64, 4, stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, 3, stride=(1, 1))

        self.conv21 = nn.Conv2d(4, 32, 8, stride=(4, 4))
        self.pool2 = nn.MaxPool2d((1, 1), (1, 1), (0, 0), (1, 1))
        # self.pool = lambda x: x

        self.conv22 = nn.Conv2d(32, 64, 4, stride=(2, 2))
        self.conv23 = nn.Conv2d(64, 64, 3, stride=(1, 1))

        self.W = torch.nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

        self.lin_in_shape = self.lin_in_shape()
        self.linear1 = nn.Linear(64 * np.prod(self.lin_in_shape), 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, self.num_actions)
        # self.batch_norm32 = nn.BatchNorm2d(32)
        self.batch_norm32_1 = nn.BatchNorm2d(32)
        self.batch_norm32_2 = nn.BatchNorm2d(32)
        # self.batch_norm64 = nn.BatchNorm2d(64)
        self.batch_norm64_1 = nn.BatchNorm2d(64)
        self.batch_norm64_2 = nn.BatchNorm2d(64)
        self.batch_norm64_3 = nn.BatchNorm2d(64)
        self.batch_norm64_4 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = torch.nn.Softmax()
        self.load_model = load_model
        self.epoch = epoch

        self.gaze_pred_model = gaze_pred_model

    def forward(self, x):
        # frame forward

        x = self.pool(self.relu(self.conv1(x)))
        x = self.batch_norm32_1(x)
        x = self.dropout(x)

        x = self.pool(self.relu(self.conv2(x)))
        x = self.batch_norm64_1(x)
        x = self.dropout(x)

        x = self.pool(self.relu(self.conv3(x)))
        x = self.batch_norm64_2(x)
        x = self.dropout(x)

        x = x.flatten(start_dim=1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

    def out_shape(self, layer, in_shape):
        h_in, w_in = in_shape
        h_out, w_out = floor(
            (
                (
                    h_in
                    + 2 * layer.padding[0]
                    - layer.dilation[0] * (layer.kernel_size[0] - 1)
                    - 1
                )
                / layer.stride[0]
            )
            + 1
        ), floor(
            (
                (
                    w_in
                    + 2 * layer.padding[1]
                    - layer.dilation[1] * (layer.kernel_size[1] - 1)
                    - 1
                )
                / layer.stride[1]
            )
            + 1
        )
        return h_out, w_out

    def lin_in_shape(self):
        # TODO create as a wrapper
        # wrapper that gives num params

        # temp written down shape calcer
        out_shape = self.out_shape(self.conv1, self.input_shape)
        out_shape = self.out_shape(self.conv2, out_shape)
        out_shape = self.out_shape(self.conv3, out_shape)
        return out_shape

    def loss_fn(self, loss_, acts, targets):
        ce_loss = loss_(acts, targets).to(device=self.device)
        return ce_loss

    def train_loop(self, opt, lr_scheduler, loss_, batch_size=32, gaze_pred=None):
        self.loss_ = loss_
        self.gaze_pred_model = gaze_pred
        if self.gaze_pred_model is not None:
            self.gaze_pred_model.eval()

        if self.load_model:
            model_pickle = torch.load(self.model_save_string.format(self.epoch))
            self.load_state_dict(model_pickle["model_state_dict"])
            opt.load_state_dict(model_pickle["model_state_dict"])
            self.epoch = model_pickle["epoch"]
            loss_val = model_pickle["loss"]
        eix = 0
        start_epoch = self.epoch
        end_epoch = self.epoch + 20
        for epoch in range(start_epoch, end_epoch):
            for i, data in enumerate(self.train_data_iter):
                x, y, _ = self.get_data(data)

                opt.zero_grad()

                acts = self.forward(x)
                loss = self.loss_fn(loss_, acts, y)
                loss.backward()
                opt.step()
                self.writer.add_scalar("Loss", loss.data.item(), eix)
                self.writer.add_scalar("Acc", self.batch_acc(acts, y), eix)

                eix += 1
            if epoch % 1 == 0:
                # self.writer.add_histogram("acts", y)
                # self.writer.add_histogram("preds", acts)
                self.writer.add_scalar("Train Acc", self.accuracy(), epoch)
                print("Epoch ", epoch, "loss", loss)
                self.calc_val_metrics(epoch)
                self.game_play(epoch)

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "loss": loss,
                    },
                    self.model_save_string.format(epoch),
                )

    def process_gaze(self, gaze):
        gaze = torch.exp(gaze)
        gazes = []
        for g in gaze:
            g = (g - torch.min(g)) / (torch.max(g) - torch.min(g))
            # g = g / torch.sum(g)
            gazes.append(g)
        gaze = torch.stack(gazes)
        del gazes
        return gaze

    def get_data(self, data):
        if isinstance(data, dict):
            x = data["images"].to(device=self.device)
            y = data["actions"].to(device=self.device)
            x_g = data["gazes"].to(device=self.device)

        elif isinstance(data, list):
            x, y, x_g = data

        return x, y, x_g

    def infer(self, x_var):
        with torch.no_grad():
            self.eval()

            acts = self.forward(x_var).argmax()
            self.train()

        return acts

    def load_model_fn(self, epoch):
        self.epoch = epoch
        model_pickle = torch.load(self.model_save_string.format(self.epoch))
        model_state_dict = {}
        for k in model_pickle["model_state_dict"]:
            if not k.__contains__("gaze_pred"):
                model_state_dict[k] = model_pickle["model_state_dict"][k]

        model_pickle["model_state_dict"] = model_state_dict

        self.load_state_dict(model_pickle["model_state_dict"])

        self.epoch = model_pickle["epoch"]
        loss_val = model_pickle["loss"]
        print(
            "Loaded {} model from saved checkpoint {} with loss {}".format(
                self.__class__.__name__, self.epoch, loss_val
            )
        )

    def batch_acc(self, acts, y):
        with torch.no_grad():
            acc = (acts.argmax(dim=1) == y).sum().data.item() / y.shape[0]
        return acc

    def calc_val_metrics(self, e):
        acc = 0
        ix = 0
        loss = 0
        self.eval()
        with torch.no_grad():
            for i, data in enumerate(self.val_data_iter):
                x, y, _ = self.get_data(data)

                acts = self.forward(x)
                loss += self.loss_fn(self.loss_, acts, y).data.item()

                acc += (acts.argmax(dim=1) == y).sum().data.item()
                ix += y.shape[0]

        self.train()
        self.writer.add_scalar("Val Loss", loss / ix, e)
        self.writer.add_scalar("Val Acc", acc / ix, e)

    def accuracy(self):
        acc = 0
        ix = 0
        self.eval()

        with torch.no_grad():
            # if True:
            for i, data in enumerate(self.train_data_iter):
                x, y, _ = self.get_data(data)
                acts = self.forward(x).argmax(dim=1)
                acc += (acts == y).sum().data.item()
                ix += y.shape[0]

        self.train()
        return acc / ix

    def game_play(self, epoch):
        transform_images = image_transforms()

        env = gym.make("Phoenix-v0", full_action_space=True, frameskip=4)
        env = FrameStack(env, 4)
        env = Monitor(env, "phoenix", force=True)

        t_rew = 0

        for i_episode in range(2):
            observation = env.reset()
            ep_rew = 0
            while True:
                observation = (
                    torch.stack([transform_images(o).squeeze() for o in observation])
                    .unsqueeze(0)
                    .to(device=self.device)
                )
                action = self.infer(observation).data.item()
                observation, reward, done, info = env.step(action)

                ep_rew += reward
                if done:
                    # print("Episode finished after {} timesteps".format(t + 1))
                    break
            t_rew += ep_rew
            print("Episode {} reward {}".format(i_episode, ep_rew))
            print("Ave Episode {} reward {}".format(i_episode, t_rew / (i_episode + 1)))
        self.writer.add_scalar("Game Reward", t_rew / (i_episode + 1), epoch)


if __name__ == "__main__":
    action_net = ACTION_SL(mode="eval")
    rand_frame = torch.rand(3, 4, 84, 84)
    rand_gaze = torch.rand(3, 4, 84, 84)
    # writer = SummaryWriter()

    action_net.writer.add_graph(action_net, input_to_model=(rand_frame, rand_gaze))
    action_net.writer.flush()
    exit()
    # action_net.writer.close()
    rand_target = torch.randint(action_net.num_actions, [1])
    action_net.forward(rand_frame, rand_gaze)
    optimizer = torch.optim.Adadelta(action_net.parameters(), lr=1.0, rho=0.95)

    # if scheduler is declared, ought to use & update it , else model never trains
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, lr_lambda=lambda x: x*0.95)
    lr_scheduler = None

    loss_ = torch.nn.CrossEntropyLoss()
    action_net.train_loop(
        optimizer, lr_scheduler, loss_, rand_frame, rand_target, rand_gaze, batch_size=4
    )
