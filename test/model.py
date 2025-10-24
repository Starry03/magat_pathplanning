import time

import lightning as pl
from torch import Tensor, stack, empty, cat, long, isnan, ones_like, max as tmax, ones
from torch.nn import (
    Conv2d,
    Linear,
    Flatten,
    ReLU,
    MaxPool2d,
    Sequential,
    BatchNorm2d,
    Dropout,
    AdaptiveAvgPool2d,
    CrossEntropyLoss,
)
import torch
from torch_geometric.nn import GCNConv, ChebConv, Sequential as GSequential
from torch.optim import Adam, lr_scheduler
from torchmetrics import Accuracy

from utils.metrics import MonitoringMultiAgentPerformance
from utils.new_simulator import multiRobotSimNew
from logger import logger


class PaperArchitecture(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.n_agents = config["num_agents"]
        self.FOV = config.get("FOV", 4)
        self.wl, self.hl = self.FOV + 2, self.FOV + 2
        self.CHANNELS: int = 3
        self.n_actions: int = 5
        self.E = 1
        self.nGraphFilterTaps = self.config["nGraphFilterTaps"]
        self.robot = multiRobotSimNew(self.config)
        self.recorder = MonitoringMultiAgentPerformance(self.config)
        self.loss = CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=self.n_actions)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.n_actions)

        # layers
        self.pool = AdaptiveAvgPool2d((1, 1))
        self.drop = Dropout(p=0.2)
        self.activation = ReLU(inplace=True)
        self.conv1 = self._conv_block(self.CHANNELS, 32)
        self.conv2 = self._conv_block(32, 64)
        self.conv3 = self._conv_block(64, 128)
        self.compress = Sequential(
            Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(128),
            self.activation,
        )
        self.flatten = Flatten()

        self.cgnn = ChebConv(
            in_channels=128,
            out_channels=128,
            K=self.nGraphFilterTaps,  # K-hop polynomial filter
            normalization="sym",  # Symmetric normalization (I - D^{-1/2} A D^{-1/2})
        )

        # Option 2: Stack of K GCN layers (previous approach)
        # self.cgnn = self._graph_block(128, 128, k=self.nGraphFilterTaps)

        self.fc = Linear(128, self.n_actions)
        self.S = ones(1, self.E, self.n_agents, self.n_agents, device=self.device)


    def _agents_to_edge_index(self, S: Tensor):
        """

        return: edge_index in shape [2, E]
        """
        B, _, N, _ = S.shape
        rows, cols = [], []
        for b in range(B):
            idx = (S[b] > 0).nonzero(as_tuple=False)  # coppie (i,j)
            if idx.numel() == 0:
                continue

            # geometric offset to have a single big graph
            rows.append(idx[:, 0] + b * N)
            cols.append(idx[:, 1] + b * N)
        if not rows:
            return empty(2, 0, dtype=long, device=S.device)

        # concatenate all batches
        return stack([cat(rows), cat(cols)], dim=0)

    def addGSO(self, S: Tensor) -> None:
        """
        Training is made on different steps of different maps so we need to get the GSO everytime

        this function refers to the original one in the repository
        """
        if S.shape == torch.Size([1, 1]):
            self.S = ones(
                1, self.E, self.n_agents, self.n_agents, device=self.device
            )
            return
        if self.E == 1:
            assert len(S.shape) == 3
            self.S = S.unsqueeze(1)
        else:
            assert len(S.shape) == 4
            assert S.shape[1] == self.E
            self.S = S

        # Remove nan data
        self.S[isnan(self.S)] = 0
        if self.config["GSO_mode"] == "dist_GSO_one":
            self.S[self.S > 0] = 1
        elif self.config["GSO_mode"] == "full_GSO":
            self.S = ones_like(self.S)
        self.S = self.S.to(self.device)

    def _conv_block(
        self,
        inp: int,
        output: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        dilation: int = 1,
        ceiling_mode: bool = False,
    ) -> Sequential:
        return Sequential(
            Conv2d(
                in_channels=inp,
                out_channels=output,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            BatchNorm2d(output, eps, momentum, affine, track_running_stats),
            self.activation,
            MaxPool2d(
                kernel_size=2,
                stride=2,
                padding=0,
                dilation=dilation,
                ceil_mode=ceiling_mode,
            ),
            Conv2d(
                in_channels=output,
                out_channels=output,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            BatchNorm2d(output, eps, momentum, affine, track_running_stats),
            self.activation,
        )

    def _graph_block(self, inp: int, output: int, k: int = 3) -> GSequential:
        """
        k graph convolutional layers with ReLU activation
        node will have info from k-hop neighbors
        """
        layers = []
        for _ in range(k):
            layers.append((GCNConv(inp, output, improved=True), "x, edge_index -> x"))
            layers.append(self.activation)
            layers.append(self.drop)
            inp = output
        return GSequential(input_args="x, edge_index", modules=layers)

    def _format_to_conv2d(self, x: Tensor) -> Tensor:
        """
        Conv2d expects (B, C, W, H), we have (B, N, C, W, H),
        so we merge B and N dimensions to avoid looping
        """
        # if (len(x.shape) == 4):
        #     return x
        (B, N, C, W, H) = x.shape
        return x.view(B * N, C, W, H)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network
        x: (B, N, C, W, H)
        return: (B, N, n_actions)
        """
        # convolutional layers
        x = self._format_to_conv2d(x)
        x = self.flatten(
            self.compress(
                self.pool(self.conv3(self.drop(self.conv2(self.drop(self.conv1(x))))))
            )
        )  # output [B*N, 128]

        # graph convolutional layer with K-hop filter taps
        edge_index = self._agents_to_edge_index(self.S)
        x = self.cgnn(x, edge_index)  # [B*N, 128]

        # fully connected layers
        x = self.activation(x)
        x = self.fc(x)  # [B*N, n_actions]

        return x

    def training_step(self, batch, batch_idx):
        batch_input, batch_target, _, batch_GSO, _ = batch
        (B, N, _, _, _) = batch_input.shape
        batch_target = batch_target.reshape(B * N, self.n_actions)
        self.addGSO(batch_GSO)
        logits = self(batch_input)
        targets = tmax(batch_target, 1)[1]
        loss = self.loss(logits, targets)
        self.train_acc(logits, targets)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_step=True, on_epoch=True
        )
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_input, batch_target, _, batch_GSO, _ = batch
        (B, N, _, _, _) = batch_input.shape
        batch_target = batch_target.reshape(B * N, self.n_actions)
        self.addGSO(batch_GSO)
        logits = self(batch_input)
        targets = tmax(batch_target, 1)[1]
        loss = self.loss(logits, targets)
        self.val_acc(logits, targets)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def mutliAgent_ActionPolicy(
        self, input, load_target, makespanTarget, tensor_map, ID_dataset, mode
    ):
        self.robot.setup(
            input, load_target, makespanTarget, tensor_map, ID_dataset, mode
        )
        maxstep = self.robot.getMaxstep()
        allReachGoal = False
        noReachGoalbyCollsionShielding = False
        check_collisionFreeSol = False
        check_CollisionHappenedinLoop = False
        check_CollisionPredictedinLoop = False
        findOptimalSolution = False
        compare_makespan, compare_flowtime = self.robot.getOptimalityMetrics()
        currentStep = 0

        Case_start = time.time()
        Time_cases_ForwardPass = []
        for step in range(maxstep):
            currentStep = step + 1
            currentState = self.robot.getCurrentState()
            currentStateGPU = currentState.to(self.config.device)

            gso = self.robot.getGSO(step)
            gsoGPU = gso.to(self.config.device)
            self.addGSO(gsoGPU)
            step_start = time.time()
            actionVec_predict = self(currentStateGPU)  # B x N X 5
            if self.config.batch_numAgent:
                actionVec_predict = actionVec_predict.detach().cpu()
            else:
                actionVec_predict = [ts.detach().cpu() for ts in actionVec_predict]
            time_ForwardPass = time.time() - step_start
            Time_cases_ForwardPass.append(time_ForwardPass)
            allReachGoal, check_moveCollision, check_predictCollision = self.robot.move(
                actionVec_predict, currentStep
            )

            if check_moveCollision:
                check_CollisionHappenedinLoop = True
            if check_predictCollision:
                check_CollisionPredictedinLoop = True
            if allReachGoal:
                break
            elif currentStep >= (maxstep):
                break

        num_agents_reachgoal = self.robot.count_numAgents_ReachGoal()
        store_GSO, store_communication_radius = self.robot.count_GSO_communcationRadius(
            currentStep
        )

        if allReachGoal and not check_CollisionHappenedinLoop:
            check_collisionFreeSol = True
            noReachGoalbyCollsionShielding = False
            findOptimalSolution, compare_makespan, compare_flowtime = (
                self.robot.checkOptimality(True)
            )
            if self.config.log_anime and self.config.mode == "test":
                self.robot.save_success_cases("success")

        if currentStep >= (maxstep):
            findOptimalSolution, compare_makespan, compare_flowtime = (
                self.robot.checkOptimality(False)
            )

        if currentStep >= (maxstep) and not allReachGoal:
            if self.config.log_anime and self.config.mode == "test":
                self.robot.save_success_cases("failure")

        if (
            currentStep >= (maxstep)
            and not allReachGoal
            and check_CollisionPredictedinLoop
            and not check_CollisionHappenedinLoop
        ):
            findOptimalSolution, compare_makespan, compare_flowtime = (
                self.robot.checkOptimality(False)
            )
            # print("### Case - {} -Step{} exceed maxstep({})- ReachGoal: {} due to CollsionShielding \n".format(ID_dataset,currentStep,maxstep, allReachGoal))
            noReachGoalbyCollsionShielding = True
            if self.config.log_anime and self.config.mode == "test":
                self.robot.save_success_cases("failure")
        time_record = time.time() - Case_start

        if self.config.mode == "test":
            exp_status = (
                "################## {} - End of loop ################## ".format(
                    self.config.exp_name
                )
            )
            case_status = "####### Case{} \t Computation time:{} \t Step{}/{}\t- AllReachGoal-{}\n".format(
                ID_dataset, time_record, currentStep, maxstep, allReachGoal
            )

            logger.info("{} \n {}".format(exp_status, case_status))

        return (
            allReachGoal,
            noReachGoalbyCollsionShielding,
            findOptimalSolution,
            check_collisionFreeSol,
            check_CollisionPredictedinLoop,
            compare_makespan,
            compare_flowtime,
            num_agents_reachgoal,
            store_GSO,
            store_communication_radius,
            time_record,
            Time_cases_ForwardPass,
        )

    def test_step(self, batch, batch_idx):
        """

        single dim batch
        """
        logger.info("test started")
        self.recorder.reset()
        (
            batch_input,
            batch_target,
            batch_makespanTarget,
            batch_tensor_map,
            batch_ID_dataset,
        ) = batch
        log_result = self.mutliAgent_ActionPolicy(
            batch_input,
            batch_target,
            batch_makespanTarget,
            batch_tensor_map,
            self.recorder.count_validset,
            mode="test",
        )
        self.recorder.update(self.robot.getMaxstep(), log_result)
        logger.info(
            "Accurracy(reachGoalnoCollision): {} \n  "
            "DeteriorationRate(MakeSpan): {} \n  "
            "DeteriorationRate(FlowTime): {} \n  "
            "Rate(collisionPredictedinLoop): {} \n  "
            "Rate(FailedReachGoalbyCollisionShielding): {} \n ".format(
                round(self.recorder.rateReachGoal, 4),
                round(self.recorder.avg_rate_deltaMP, 4),
                round(self.recorder.avg_rate_deltaFT, 4),
                round(self.recorder.rateCollisionPredictedinLoop, 4),
                round(self.recorder.rateFailedReachGoalSH, 4),
            )
        )
        return self.recorder.rateReachGoal

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.config.get("learning_rate", 1e-3),
            weight_decay=self.config.get("weight_decay", 0),
        )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.config.get("max_epochs", 10),
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
