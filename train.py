from datetime import datetime
import os
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import csv

from save_load import save_model as save_model_default
from utils.move_to_device import move_to_device


GET_DEFAULT_OPTIMIZER = lambda m: torch.optim.Adam(m.parameters(), lr=1e-3)
DEFAULT_CRITERION = torch.nn.MSELoss()

class Trainer:
    def __init__(self, model, evaluate,
                 optimizer=None, criterion=None, scheduler=None,
                 experiment_name: str = 'model',
                 save_model = save_model_default,
                 existing_path = None):
        """ Initialzer """
        self.save_model = save_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optimizer if optimizer else GET_DEFAULT_OPTIMIZER(model)
        self.criterion = criterion if criterion else DEFAULT_CRITERION
        self.scheduler = scheduler
        self.model = model
        self.model.to(self.device)
        self.evaluate = evaluate
        self.csv_headers = []

        self.best_model = None
        self.best_optimizer = None
        self.best_scheduler = None
        self.best_epoch = 0
        self.best_loss = 1e10 # best mean loss

        self._setup_experiment(experiment_name, existing_path)

        return

    def _setup_experiment(self, experiment_name, existing_path):
        """ Sets up folders for experiments """
        save_root = 'experiments'
        timestamp = existing_path if existing_path is not None else datetime.now().strftime("%d-%m-%Y_%H-%M")
        exp_dir = os.path.join(save_root, f"{timestamp}_{experiment_name}")
        os.makedirs(exp_dir, exist_ok=True)
        # tensorboard folder
        dir_tboard = os.path.join(exp_dir, 'tboard')
        os.makedirs(dir_tboard, exist_ok=True)
        self.writer = SummaryWriter(dir_tboard)
        # folder for image logs
        dir_imgs = os.path.join(exp_dir, 'imgs')
        os.makedirs(dir_imgs, exist_ok=True)
        self.dir_imgs = dir_imgs
        # model snapshots
        dir_checkpoints = os.path.join(exp_dir, 'checkpoints')
        os.makedirs(dir_checkpoints, exist_ok=True)
        self.dir_checkpoints = dir_checkpoints

        dir_logs = os.path.join(exp_dir, 'logs')
        os.makedirs(dir_logs, exist_ok=True)
        self.file_metrics = os.path.join(dir_logs, 'metrics.csv')
        self.file_logs = os.path.join(dir_logs, 'logs.txt') # console output saves
        logs_exists = os.path.isfile(self.file_logs)
        if not logs_exists: open(self.file_logs, "x")
        
        with open(os.path.join(dir_logs, 'model_architecture.txt'), "w") as f:
            print(self.model, file=f)
        with open(os.path.join(dir_logs, 'optimizer_architecture.txt'), "w") as f:
            print(self.optimizer, file=f)
        self.file_params = os.path.join(dir_logs, 'hyperparameters.txt')

        return
        
    def _log_to_csv(self, data, headers):
        with open(self.file_metrics, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            
            if f.tell() == 0:
                writer.writeheader()
            
            writer.writerow(data)
    
    def _log_params(self, text: str):
        """ Save training hyperparameters information """
        with open(self.file_params, 'a') as f:
            f.write(text + '\n')
    
    def _log(self, info):
        print(info)
        with open(self.file_logs, 'a') as f:
            f.write(str(info) + '\n')
    
    # =========================== TRAINING LOGIC ===========================

    def train_one_step(self, inputs):
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(inputs)
        loss = self.criterion(outputs, inputs)
        loss.backward()
        self.optimizer.step()

        loss_item = loss.detach().cpu().item()
        
        # free intermediate tensors early
        del outputs, loss

        return loss_item

    
    def train(self, train_loader, val_loader, epochs=10, init_step=0, eval_freq=1e3, save_freq=None):
        """ Training the models for several iterations """
        total_batches = len(train_loader)
        iter_ = init_step
        init_epoch = (iter_) // total_batches
        batch_size = train_loader.batch_size

        EVAL_FREQUENCY = eval_freq # how often to evaluate
        SAVE_FREQUENCY = save_freq if save_freq else total_batches - 1 # how often to take snapshots
        
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']
        self._log_params(f"epochs={epochs}, batch_size={batch_size}, learning_rate={get_lr(self.optimizer)}, start_step={init_step}")

        progress_bar = tqdm(total=epochs, initial=init_step)

        for epoch in range(init_epoch, init_epoch+epochs):
            loss_list = []
            mean_loss = .0

            for batch in train_loader:
                batch = move_to_device(batch, self.device)
                
                loss_item = self.train_one_step(batch)
                loss_list.append(loss_item)
            
                # updating progress bar
                progress_bar.set_description(f"Ep {epoch} Iter {iter_}: Loss={round(loss_item,5)})")
                
                # logging
                self.writer.add_scalar(f'Loss/Train', loss_item, global_step=iter_)

                train_metrics = {
                    "iter": iter_,
                    "epoch": epoch,
                    "loss": loss_item
                }

                # if last batch of the epoch, track averaged loss as well
                is_last_batch = epoch > 0 and ((iter_+1) // total_batches) > epoch
                if is_last_batch:
                    mean_loss = np.mean(loss_list)
                    train_metrics["mean_loss"] = mean_loss
                    # track the best configuration
                    if mean_loss < self.best_loss:
                        self.best_model = self.model.state_dict()
                        self.best_optimizer = self.optimizer.state_dict()
                        if self.scheduler: self.best_scheduler = self.scheduler.state_dict()
                        self.best_loss = mean_loss
                        self.best_epoch = epoch
                
                
                csv_headers = ["iter", "epoch", "loss", "mean_loss"]
                # EVALUATION STEP
                if ((iter_ % EVAL_FREQUENCY == 0 or is_last_batch) and self.evaluate is not None):
                    # evaluation metrics
                    eval_metrics = self.evaluate(self.model, val_loader, self.device)
                    self.model.train()

                    assert isinstance(eval_metrics, dict), "Eval metrics must be of dict type for CSV logging."
                    eval_metric_names = eval_metrics.keys()

                    csv_headers += eval_metric_names
                    metrics = {**train_metrics, **eval_metrics}
                    self._log(metrics)
                    self._log_to_csv(metrics, csv_headers)

                    for metric_name in eval_metric_names:
                        self.writer.add_scalar(f"{metric_name}/Valid", eval_metrics[metric_name], global_step=iter_)

                    # image logging
                    if (iter_ > 0):
                        with torch.no_grad():
                            self.model.eval()
                            recon = self.model(batch)

                            if isinstance(recon, (tuple, list)):
                                recon = recon[0]

                            if recon.ndim == 5:
                                recon = recon[:, 0] # take batch with T=1 for logging
                            grid = torchvision.utils.make_grid(recon.detach().cpu())
                            self.writer.add_image('Images/Train', grid, global_step=iter_)
                            torchvision.utils.save_image(grid, os.path.join(self.dir_imgs, f"imgs_{iter_}.png"))
                            del recon, grid

                            self.model.train()
                    
                # o/w track only train metrics
                else:
                    self._log_to_csv(train_metrics, csv_headers)

                
                iter_ = iter_ + 1

            # SAVE SNAPSHOT
            # ! too costly feature
            finished_epoch = (iter_) // total_batches
            if (finished_epoch > 0) and (finished_epoch % SAVE_FREQUENCY == 0):
                self.save_model(self.model, self.optimizer, self.scheduler,
                            stats={ "epoch": finished_epoch, "iter_": iter_ },
                            save_path=self.dir_checkpoints,
                            model_name=f"epoch_{finished_epoch:03d}_iter_{iter_:05d}")
            
            if self.scheduler: self.scheduler.step(mean_loss)

        print(f"Training completed")

        # save very last state
        finished_epoch = (iter_) // total_batches
        self.save_model(self.model, None, self.scheduler,
                    stats={ "epoch": finished_epoch, "iter_": iter_ },
                    save_path=self.dir_checkpoints,
                    model_name=f"epoch_{finished_epoch:03d}_iter_{iter_:05d}")
        
        # save the best model condition
        # torch.save({
        #     'model_state_dict': self.best_model,
        #     'optimizer_state_dict': self.best_optimizer,
        #     'scheduler_state_dict': self.best_scheduler,
        #     'stats': { "epoch": self.best_epoch }
        # }, f"{self.dir_checkpoints}/best_model_{self.best_epoch:03d}.pth")

        return


class TrainerAR(Trainer):
    """
    Trainer for autoregressive VideoARTransformer (target -- rgb)
    (teacher forcing)
    """
    def __init__(self, model, evaluate,
                 optimizer=None, criterion=None, scheduler=None,
                 experiment_name: str = 'ar_model',
                 save_model=None):
        super().__init__(model, evaluate, optimizer, criterion, scheduler,
                         experiment_name, save_model if save_model else save_model_default)

    def train_one_step(self, batch):
        """
        batch 
          ground truth  -> (B, 9, C, H, W)  (teacher-forced seq)
          targets are the last 5 elements -> (B, 5, C, H, W) 
        """
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(batch)  # (B, 9, C, H, W) with predictions ŷ₂..ŷ₁₀

        # take the last 5 frames from outputs (positions [5..9] → predictions for [6..10])
        preds_last5 = outputs[:, -5:] # (B, 5, C, H, W)
        targets_last5 = batch[:, -5:] # (B, 5, C, H, W)

        loss = self.criterion(preds_last5, targets_last5)
        loss.backward()
        self.optimizer.step()

        loss_item = loss.detach().cpu().item()
        del outputs, preds_last5, targets_last5, loss

        return loss_item

    def _log_predictions(self, inputs, targets, outputs, step, tag="Train"):
        """
        Логгирование картинок в TensorBoard и в imgs/
        """
        B = inputs.size(0)
        # берём только первый элемент в батче для логирования
        inp_seq = inputs[0]    # (9, C, H, W)
        tgt_seq = targets[0]   # (5, C, H, W)
        out_seq = outputs[0, -5:]  # (5, C, H, W)

        # соберём картинки рядом: входы | ground truth | предсказания
        grid = torch.cat([
            inp_seq,                          # 9 кадров
            torch.zeros_like(inp_seq[:1]),    # разделитель
            tgt_seq,                          # 5 GT
            out_seq                           # 5 preds
        ], dim=0)  # (N, C, H, W)

        grid = torchvision.utils.make_grid(grid, nrow=7, normalize=True)
        self.writer.add_image(f"Sequences/{tag}", grid, global_step=step)

        torchvision.utils.save_image(
            grid, os.path.join(self.dir_imgs, f"{tag.lower()}_{step:06d}.png")
        )

    def train(self, train_loader, val_loader, epochs=10, eval_freq=1000, save_freq=None):
        iter_ = 0
        total_batches = len(train_loader)
        progress_bar = tqdm(total=epochs)

        for epoch in range(epochs):
            loss_list = []
            mean_loss = .0

            for batch in train_loader:
                batch = move_to_device(batch, self.device)
                inputs, targets = batch

                loss_item = self.train_one_step(batch)
                loss_list.append(loss_item)

                progress_bar.set_description(f"Ep {epoch} Iter {iter_}: Loss={round(loss_item,5)})")
                self.writer.add_scalar('Loss/Train', loss_item, global_step=iter_)

                # логируем последовательности каждые 500 итераций
                if iter_ % 500 == 0:
                    with torch.no_grad():
                        self.model.eval()
                        outputs = self.model(inputs)
                        self._log_predictions(inputs, targets, outputs, step=iter_, tag="Train")
                        self.model.train()

                iter_ += 1

            mean_loss = np.mean(loss_list)
            if self.scheduler: 
                self.scheduler.step(mean_loss)

        print("Training completed")
