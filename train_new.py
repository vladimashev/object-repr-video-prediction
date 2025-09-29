from datetime import datetime
import os
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import csv
import imageio
from PIL import Image

from save_load import save_model as save_model_default
from utils.move_to_device import move_to_device
from utils.visualize_video_sample import draw_rollout_grid, masks_to_rgb_tensor


GET_DEFAULT_OPTIMIZER = lambda m: torch.optim.Adam(m.parameters(), lr=1e-3)
DEFAULT_CRITERION = torch.nn.MSELoss()

class Trainer:
    def __init__(self, model, evaluate,
                 optimizer=None, criterion=None, scheduler=None,
                 experiment_name: str = 'model',
                 save_model = save_model_default):
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

        self._setup_experiment(experiment_name)

        return

    def _setup_experiment(self, experiment_name):
        """ Sets up folders for experiments """
        save_root = 'experiments'
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
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

        return

    def _log_predictions(self, inputs, iter_):
        self.model.eval()
        recon = self.model(batch)
        grid = torchvision.utils.make_grid(preds.detach().cpu())
        self.writer.add_image('Images/Train', grid, global_step=iter_)
        torchvision.utils.save_image(grid, os.path.join(self.dir_imgs, f"imgs_{iter_}.png"))
        del grid, recon
        self.model.train()
       
        
    def _log_to_csv(self, data, headers):
        with open(self.file_metrics, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            
            if f.tell() == 0:
                writer.writeheader()
            
            writer.writerow(data)
    
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
        iter_ = 0
        total_batches = len(train_loader)
        batch_size = train_loader.batch_size

        EVAL_FREQUENCY = eval_freq # how often to evaluate
        SAVE_FREQUENCY = save_freq if save_freq else total_batches - 1 # how often to take snapshots
        
        progress_bar = tqdm(total=epochs, initial=init_step)

        for epoch in range(epochs):
            loss_list = []
            mean_loss = .0

            for batch in train_loader:
                batch = move_to_device(batch, self.device)
                
                loss_item = self.train_one_step(batch, iter_)
                loss_list.append(loss_item)
            
                # updating progress bar
                progress_bar.set_description(f"Ep {epoch} Iter {iter_}: Loss={round(loss_item,5)})")
                
                # logging
                self.writer.add_scalar(f'Loss/Train', loss_item, global_step=iter_)

                train_metrics = {
                    "iter": iter_,
                    "batch_size": batch_size,
                    "epoch": epoch,
                    "loss": loss_item
                }

                # if last batch of the epoch, track averaged loss as well
                if (epoch > 0 and ((iter_+1) // total_batches) > epoch):
                    mean_loss = np.mean(loss_list)
                    train_metrics["mean_loss"] = mean_loss
                    # track the best configuration
                    if mean_loss < self.best_loss:
                        self.best_model = self.model.state_dict()
                        self.best_optimizer = self.optimizer.state_dict()
                        if self.scheduler: self.best_scheduler = self.scheduler.state_dict()
                        self.best_loss = mean_loss
                        self.best_epoch = epoch
                
                csv_headers = ["iter", "batch_size", "epoch", "loss", "mean_loss"]
                
                # EVALUATION STEP
                if (iter_ % EVAL_FREQUENCY == 0 and self.evaluate is not None):
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
                            val_batch = move_to_device(next(iter(val_loader)), self.device)
                            self._log_predictions(val_batch, iter_)
                    
                # o/w track only train metrics
                else:
                    self._log_to_csv(train_metrics, csv_headers)

                # SAVE SNAPSHOT
                if (iter_ > 0) and (iter_ % SAVE_FREQUENCY == 0):
                    finished_epoch = (iter_+1) // total_batches
                    self.save_model(self.model, self.optimizer, self.scheduler,
                               stats={ "epoch": finished_epoch, "iter_": iter_+1 },
                               save_path=self.dir_checkpoints,
                               model_name=f"epoch_{finished_epoch:03d}_iter_{iter_+1:05d}")
                
                iter_ = iter_ + 1
            
            if self.scheduler: 
                self.scheduler.step(mean_loss)

        print(f"Training completed")

        # save very last state
        finished_epoch = (iter_) // total_batches
        self.save_model(self.model, self.optimizer, self.scheduler,
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


class TrainerRGBTeacherForce(Trainer):
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

    def train_one_step(self, inputs):
        """
        batch 
          input  -> (B, 9, C, H, W)  (teacher-forced)
          targets are the last 5 elements -> (B, 5, C, H, W) 

          inputs              [1, 2, 3, 4, 5, 6, 7, 8, 9] 
          model yields prds   [_, _, _, _, 5, 6, 7, 8, 9] for [6, 7, 8, 9, 10]
          targets             [_, _, _, _, 6, 7, 8, 9, 10]

        """
        self.model.train()
        self.optimizer.zero_grad()

        outputs = self.model(inputs[:, :-1, ...].contiguous())  # (B, 9, C, H, W) predictions for frames [6..10] at positions [5..9]

        # take the last 5 frames from outputs (positions [5..9] → predictions for [6..10])
        preds_last5 = outputs[:, -5:] # (B, 5, C, H, W)
        targets_last5 = inputs[:, -5:] # (B, 5, C, H, W)

        loss = self.criterion(preds_last5, targets_last5)
        loss.backward()
        self.optimizer.step()

        loss_item = loss.detach().cpu().item()
        del outputs, preds_last5, targets_last5, loss

        return loss_item

    def _log_predictions(self, batch, iter_):
        context = batch[0, :5].unsqueeze(0)   # (1, 5, C, H, W)
        future  = batch[0, 5:].unsqueeze(0)   # (1, 15, C, H, W)
        B, _, C, H, W = context.shape
    
        preds = []
        cur_context = context.clone()
        with torch.no_grad():
            for t in range(15):
                out = self.model(cur_context)             # [1, T=5, C, H, W]
                next_frame = out[:, -1]                   # берём последний предсказанный кадр
                preds.append(next_frame.unsqueeze(1))     # [1, 1, C, H, W]
                # autoregressive update
                cur_context = torch.cat([cur_context[:, 1:], next_frame.unsqueeze(1)], dim=1)
    
        preds = torch.cat(preds, dim=1)  # (1, 15, C, H, W)
    
        ctx_seq   = context[0]      # (5, C, H, W)
        fut_seq   = future[0]       # (15, C, H, W)
        pred_seq  = preds[0]        # (15, C, H, W)
    
        # Грид: входы | разделитель | GT | предсказания
        grid = torch.cat([
            ctx_seq,
            torch.zeros_like(ctx_seq[:1]),   # разделитель
            fut_seq,
            pred_seq
        ], dim=0)  # (N, C, H, W)
    
        grid = torchvision.utils.make_grid(grid, nrow=7, normalize=True)
    
        self.writer.add_image(f"Sequences/rollout", grid, global_step=iter_)
        torchvision.utils.save_image(
            grid, os.path.join(self.dir_imgs, f"rollout_{iter_:06d}.png")
        )

        # gif
        scale = 4
        seq_for_gif = torch.cat([ctx_seq, pred_seq], dim=0) 
        seq_for_gif = (seq_for_gif.clamp(0,1) * 255).byte().cpu()  # [T, C, H, W]
        seq_for_gif = seq_for_gif.permute(0, 2, 3, 1).numpy()      # [T, H, W, C]

        images = []
        for f in seq_for_gif:
            img = Image.fromarray(f)
            if scale != 1.0:
                w, h = img.size
                img = img.resize((int(w * scale), int(h * scale)), Image.NEAREST)
            images.append(img)

        gif_path = os.path.join(self.dir_imgs, f"rollout_{iter_:06d}.gif")
        imageio.mimsave(gif_path, images, fps=5)


class TrainerObjTeacherForce(Trainer):
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

    def train_one_step(self, inputs):
        """
        batch 
          input  -> (B, 9, C, H, W)  (teacher-forced)
          targets are the last 5 elements -> (B, 5, C, H, W) 

          inputs              [1, 2, 3, 4, 5, 6, 7, 8, 9] 
          model yields prds   [_, _, _, _, 5, 6, 7, 8, 9] for [6, 7, 8, 9, 10]
          targets             [_, _, _, _, 6, 7, 8, 9, 10]

        """
        self.model.train()
        self.optimizer.zero_grad()

        outputs = self.model(inputs[:, :-1, ...].contiguous())  # (B, 9, C, H, W) predictions for frames [6..10] at positions [5..9]

        # take the last 5 frames from outputs (positions [5..9] → predictions for [6..10])
        preds_last5 = outputs[:, -5:] # (B, 5, C, H, W)
        targets_last5 = inputs[:, -5:] # (B, 5, C, H, W)

        loss = self.criterion(preds_last5, targets_last5)
        loss.backward()
        self.optimizer.step()

        loss_item = loss.detach().cpu().item()
        del outputs, preds_last5, targets_last5, loss

        return loss_item

    def _log_predictions(self, batch, iter_):
        frames, masks = batch
        # берём первый элемент батча для наглядности
        context = (frames[0:1, :5], masks[0:1, :5])   # ([1,5,C,H,W], [1,5,H,W])
        future_f = frames[0:1, 5:20]                  # [1,15,C,H,W]   (если у вас 15 future)
        future_m = masks[0:1,  5:20]                  # [1,15,H,W]
    
        preds_f, preds_m = [], []
        cur_context = (context[0].clone(), context[1].clone())
    
        self.model.eval()
        with torch.no_grad():
            for _ in range(future_f.size(1)):
                out_f, out_m, _ = self.model(cur_context)       # ([1,5,C,H,W], [1,5,H,W])
                next_f = out_f[:, -1]                        # [1,C,H,W]
                next_m = out_m[:, -1]                        # [1,H,W]
                preds_f.append(next_f.unsqueeze(1))          # [1,1,C,H,W]
                preds_m.append(next_m.unsqueeze(1))          # [1,1,H,W]
                # autoregressive update
                cur_context = (
                    torch.cat([cur_context[0][:, 1:], next_f.unsqueeze(1)], dim=1),
                    torch.cat([cur_context[1][:, 1:], next_m.unsqueeze(1)], dim=1),
                )
    
        preds_f = torch.cat(preds_f, dim=1)[0]  # [15,C,H,W]
        preds_m = torch.cat(preds_m, dim=1)[0]  # [15,H,W]
        ctx_f, fut_f = context[0][0], future_f[0]  # [5,C,H,W], [15,C,H,W]
        ctx_m, fut_m = context[1][0], future_m[0]  # [5,H,W],   [15,H,W]
    
        # ---- Рисуем КАДРЫ (как раньше) ----
        draw_rollout_grid(ctx_f, fut_f, preds_f, self.writer, self.dir_imgs, iter_, mode='val')
    
        # ---- Рисуем МАСКИ (цветные) ----
        ctx_m_rgb  = masks_to_rgb_tensor(ctx_m)   # [5,3,H,W]
        fut_m_rgb  = masks_to_rgb_tensor(fut_m)   # [15,3,H,W]
        preds_m_rgb= masks_to_rgb_tensor(preds_m) # [15,3,H,W]
        draw_rollout_grid(ctx_m_rgb, fut_m_rgb, preds_m_rgb, self.writer, self.dir_imgs, iter_, mode='val_masks')
    
        # ---- GIF по кадрам ----
        scale = 4
        seq_for_gif = torch.cat([ctx_f, preds_f], dim=0).clamp(0,1)   # [T, C, H, W]
        seq_np = (seq_for_gif * 255).byte().cpu().permute(0,2,3,1).numpy()
        images = []
        for f in seq_np:
            img = Image.fromarray(f)
            if scale != 1.0:
                w, h = img.size
                img = img.resize((int(w*scale), int(h*scale)), Image.NEAREST)
            images.append(img)
        gif_path = os.path.join(self.dir_imgs, f"rollout_val_{iter_:06d}.gif")
        imageio.mimsave(gif_path, images, fps=5)
    
        # ---- GIF по маскам (цветные) ----
        seq_m_np = (torch.cat([ctx_m_rgb, preds_m_rgb], dim=0)
                    .clamp(0,1).mul(255).byte().cpu().permute(0,2,3,1).numpy())
        images_m = []
        for f in seq_m_np:
            img = Image.fromarray(f)
            if scale != 1.0:
                w, h = img.size
                img = img.resize((int(w*scale), int(h*scale)), Image.NEAREST)
            images_m.append(img)
        gif_path_m = os.path.join(self.dir_imgs, f"rollout_val_masks_{iter_:06d}.gif")
        imageio.mimsave(gif_path_m, images_m, fps=5)

    def _log_predictions_next5(self, batch, iter_, step=None):
        frames, masks = batch
        context = (frames[0:1, :5], masks[0:1, :5])  # ([1,5,C,H,W], [1,5,H,W])
        future_f = frames[0:1, 5:10]                 # [1,5,C,H,W]
        future_m = masks[0:1,  5:10]                 # [1,5,H,W]
    
        preds_f, preds_m = [], []
        cur_context = (context[0].clone(), context[1].clone())
    
        self.model.eval()
        with torch.no_grad():
            for t in range(5):
                out_f, out_m, _ = self.model(cur_context)      # ([1,5,C,H,W], [1,5,H,W])
                next_f = out_f[:, -1]                       # [1,C,H,W]
                next_m = out_m[:, -1]                       # [1,H,W]
                preds_f.append(next_f.unsqueeze(1))
                preds_m.append(next_m.unsqueeze(1))
    
                # по-кадровый лог (frames)
                draw_rollout_grid(
                    cur_context[0][0],                      # ctx frames: [5,C,H,W]
                    future_f[0, t, :].unsqueeze(0),        # gt frame:   [1,C,H,W]
                    next_f,                                 # pred frame: [1,C,H,W]
                    self.writer, self.dir_imgs, iter_, mode=f"train_step{t:02d}"
                )
                # по-кадровый лог (masks)
                draw_rollout_grid(
                    masks_to_rgb_tensor(cur_context[1][0]),                 # [5,3,H,W]
                    masks_to_rgb_tensor(future_m[0, t, ...].unsqueeze(0)),  # [1,3,H,W]
                    masks_to_rgb_tensor(next_m),                            # [1,3,H,W]  <-- без squeeze
                    self.writer, self.dir_imgs, iter_, mode=f"train_masks_step{t:02d}"
                )
                    
                # autoregressive update
                cur_context = (
                    torch.cat([cur_context[0][:, 1:], next_f.unsqueeze(1)], dim=1),
                    torch.cat([cur_context[1][:, 1:], next_m.unsqueeze(1)], dim=1),
                )
    
        preds_f = torch.cat(preds_f, dim=1)[0]  # [5,C,H,W]
        preds_m = torch.cat(preds_m, dim=1)[0]  # [5,H,W]
        ctx_f, fut_f = context[0][0], future_f[0]
        ctx_m, fut_m = context[1][0], future_m[0]
    
        # итоговые гриды: frames + masks
        draw_rollout_grid(ctx_f, fut_f, preds_f, self.writer, self.dir_imgs, iter_, mode='train')
        draw_rollout_grid(masks_to_rgb_tensor(ctx_m),
                          masks_to_rgb_tensor(fut_m),
                          masks_to_rgb_tensor(preds_m),
                          self.writer, self.dir_imgs, iter_, mode='train_masks')

        # GIF: frames
        scale = 4
        seq_np = (torch.cat([ctx_f, preds_f], dim=0).clamp(0,1)*255).byte().cpu().permute(0,2,3,1).numpy()
        images = []
        for f in seq_np:
            img = Image.fromarray(f)
            if scale != 1.0:
                w, h = img.size
                img = img.resize((int(w*scale), int(h*scale)), Image.NEAREST)
            images.append(img)
        gif_path = os.path.join(self.dir_imgs, f"rollout_train_{iter_:06d}.gif")
        imageio.mimsave(gif_path, images, fps=5)
    
        # GIF: masks
        seq_m_np = (torch.cat([masks_to_rgb_tensor(ctx_m), masks_to_rgb_tensor(preds_m)], dim=0)
                    .clamp(0,1).mul(255).byte().cpu().permute(0,2,3,1).numpy())
        images_m = []
        for f in seq_m_np:
            img = Image.fromarray(f)
            if scale != 1.0:
                w, h = img.size
                img = img.resize((int(w*scale), int(h*scale)), Image.NEAREST)
            images_m.append(img)
        gif_path_m = os.path.join(self.dir_imgs, f"rollout_train_masks_{iter_:06d}.gif")
        imageio.mimsave(gif_path_m, images_m, fps=5)


class TrainerAutoRegressive(Trainer):
    """
    Trainer for autoregressive VideoARTransformer (target -- rgb)
    """
    def __init__(self, model, evaluate,
                 optimizer=None, criterion=None, scheduler=None,
                 experiment_name: str = 'ar_model',
                 save_model=None):
        super().__init__(model, evaluate, optimizer, criterion, scheduler,
                         experiment_name, save_model if save_model else save_model_default)

    def train_one_step(self, inputs, iter_):
        """
        batch 
          input (B, 9, C, H, W)
          targets are the last 5 elements -> (B, 5, C, H, W) 

          inputs              [1, 2, 3, 4, 5, 6, 7, 8, 9] 
          model yields prds   [_, _, _, _, 5, 6, 7, 8, 9] for [6, 7, 8, 9, 10]
          targets             [_, _, _, _, 6, 7, 8, 9, 10]

        """
        self.model.train()
        self.optimizer.zero_grad()
        log_pred_freq = 500
        context, future = inputs[:, :5], inputs[:, 5:]  # [B,5,..], [B,5,..]
        steps = 5
        preds = []
        cur_context = context.clone()
        for t in range(steps):
            out = self.model(cur_context)          # (B, T, C, H, W)
            next_frame = out[:, -1]                # последний кадр
            preds.append(next_frame.unsqueeze(1))  # (B, 1, C, H, W)

            if iter_ % log_pred_freq == 0:
                with torch.no_grad():
                    self._log_predictions_next5(inputs, iter_)
                
            # обновляем контекст (без градиентов)
            cur_context = torch.cat(
                [cur_context[:,1:], next_frame.unsqueeze(1)],
                dim=1
            )
        
        preds = torch.cat(preds, dim=1)  # (B, 5, C, H, W)
        loss = self.criterion(preds, future)
        loss.backward()
        self.optimizer.step()

        loss_item = loss.detach().cpu().item()
        del preds, future, cur_context, loss

        return loss_item

    def _log_predictions(self, batch, iter_):
        context = batch[0, :5].unsqueeze(0)   # (1, 5, C, H, W)
        future  = batch[0, 5:].unsqueeze(0)   # (1, 15, C, H, W)
        B, _, C, H, W = context.shape
    
        preds = []
        cur_context = context.clone()
        with torch.no_grad():
            for t in range(15):
                out = self.model(cur_context)             # [1, T=5, C, H, W]
                next_frame = out[:, -1]                   # берём последний предсказанный кадр
                preds.append(next_frame.unsqueeze(1))     # [1, 1, C, H, W]
                # autoregressive update
                cur_context = torch.cat([cur_context[:, 1:], next_frame.unsqueeze(1)], dim=1)
    
        preds = torch.cat(preds, dim=1)  # (1, 15, C, H, W)
    
        ctx_seq   = context[0]      # (5, C, H, W)
        fut_seq   = future[0]       # (15, C, H, W)
        pred_seq  = preds[0]        # (15, C, H, W)
    
        draw_rollout_grid(ctx_seq, fut_seq, pred_seq, self.writer, self.dir_imgs, iter_, mode='val')

        # gif
        scale = 4
        seq_for_gif = torch.cat([ctx_seq, pred_seq], dim=0) 
        seq_for_gif = (seq_for_gif.clamp(0,1) * 255).byte().cpu()  # [T, C, H, W]
        seq_for_gif = seq_for_gif.permute(0, 2, 3, 1).numpy()      # [T, H, W, C]

        images = []
        for f in seq_for_gif:
            img = Image.fromarray(f)
            if scale != 1.0:
                w, h = img.size
                img = img.resize((int(w * scale), int(h * scale)), Image.NEAREST)
            images.append(img)

        gif_path = os.path.join(self.dir_imgs, f"rollout_val_{iter_:06d}.gif")
        imageio.mimsave(gif_path, images, fps=5)

    def _log_predictions_next5(self, batch, iter_, step=None):
        context = batch[0, :5].unsqueeze(0)   # (1, 5, C, H, W)
        future  = batch[0, 5:].unsqueeze(0)   # (1, 5, C, H, W)
        B, _, C, H, W = context.shape
    
        preds = []
        cur_context = context.clone()
        with torch.no_grad():
            for t in range(5):
                out = self.model(cur_context)             # [1, T=5, C, H, W]
                next_frame = out[:, -1]                   # берём последний предсказанный кадр
                preds.append(next_frame.unsqueeze(1))     # [1, 1, C, H, W]
                draw_rollout_grid(cur_context[0], future[0, t, :].unsqueeze(0), next_frame, self.writer, self.dir_imgs, iter_, step=t)
                # autoregressive update
                cur_context = torch.cat([cur_context[:, 1:], next_frame.unsqueeze(1)], dim=1)
    
        preds = torch.cat(preds, dim=1)  # (1, 5, C, H, W)
    
        ctx_seq   = context[0]      # (5, C, H, W)
        fut_seq   = future[0]       # (5, C, H, W)
        pred_seq  = preds[0]        # (5, C, H, W)
    
        draw_rollout_grid(ctx_seq, fut_seq, pred_seq, self.writer, self.dir_imgs, iter_)

        # gif
        scale = 4
        seq_for_gif = torch.cat([ctx_seq, pred_seq], dim=0) 
        seq_for_gif = (seq_for_gif.clamp(0,1) * 255).byte().cpu()  # [T, C, H, W]
        seq_for_gif = seq_for_gif.permute(0, 2, 3, 1).numpy()      # [T, H, W, C]

        images = []
        for f in seq_for_gif:
            img = Image.fromarray(f)
            if scale != 1.0:
                w, h = img.size
                img = img.resize((int(w * scale), int(h * scale)), Image.NEAREST)
            images.append(img)
            
        gif_path = os.path.join(self.dir_imgs, f"rollout_train_{iter_:06d}.gif")
        imageio.mimsave(gif_path, images, fps=5)


class TrainerObjAutoRegressive(Trainer):
    """
    Trainer for autoregressive VideoARTransformer (target -- obj)
    """
    def __init__(self, model, evaluate,
                 optimizer=None, criterion=None, scheduler=None,
                 experiment_name: str = 'ar_model',
                 save_model=None):
        super().__init__(model, evaluate, optimizer, criterion, scheduler,
                         experiment_name, save_model if save_model else save_model_default)

    def train_one_step(self, inputs, iter_):
        """
        tbd

        """
        self.model.train()
        self.optimizer.zero_grad()
        log_pred_freq = 500
        context, future = (inputs[0][:, :5], inputs[1][:, :5]), (inputs[0][:, 5:], inputs[1][:, 5:]) #(inputs[0][:, 5:], inputs[0][:, 5:]) for future
        steps = 5
        preds = []
        logits = []
        cur_context = (context[0].clone(), context[1].clone())
        for t in range(steps):
            out = self.model(cur_context)          # ([B, T, C, H, W)], [B, T, H, W)], [B,T,K,H,W]) of (frames, masks, masks_logits)
            next_frame = out[0][:, -1]
            next_mask = out[1][:, -1]
            preds.append(next_frame.unsqueeze(1))  # (B, 1, C, H, W)
            logits.append(out[2][:, -1].unsqueeze(1))
            if iter_ % log_pred_freq == 0:
                with torch.no_grad():
                    self._log_predictions_next5(inputs, iter_)
                
            # обновляем контекст
            cur_context = (torch.cat([cur_context[0][:, 1:], next_frame.unsqueeze(1)], dim=1),
                           torch.cat([cur_context[1][:, 1:], next_mask.unsqueeze(1)],  dim=1))

        
        preds = torch.cat(preds, dim=1)  # (B, 5, C, H, W)
        logits = torch.cat(logits, dim=1) # [B, 5, K, H, W]
        loss = self.criterion((preds, logits), future)
        loss.backward()
        self.optimizer.step()

        loss_item = loss.detach().cpu().item()
        del preds, logits, future, cur_context, loss

        return loss_item

    def _log_predictions(self, batch, iter_):
        frames, masks = batch
        # берём первый элемент батча для наглядности
        context = (frames[0:1, :5], masks[0:1, :5])   # ([1,5,C,H,W], [1,5,H,W])
        future_f = frames[0:1, 5:20]                  # [1,15,C,H,W]   (если у вас 15 future)
        future_m = masks[0:1,  5:20]                  # [1,15,H,W]
    
        preds_f, preds_m = [], []
        cur_context = (context[0].clone(), context[1].clone())
    
        self.model.eval()
        with torch.no_grad():
            for _ in range(future_f.size(1)):
                out_f, out_m, _ = self.model(cur_context)       # ([1,5,C,H,W], [1,5,H,W])
                next_f = out_f[:, -1]                        # [1,C,H,W]
                next_m = out_m[:, -1]                        # [1,H,W]
                preds_f.append(next_f.unsqueeze(1))          # [1,1,C,H,W]
                preds_m.append(next_m.unsqueeze(1))          # [1,1,H,W]
                # autoregressive update
                cur_context = (
                    torch.cat([cur_context[0][:, 1:], next_f.unsqueeze(1)], dim=1),
                    torch.cat([cur_context[1][:, 1:], next_m.unsqueeze(1)], dim=1),
                )
    
        preds_f = torch.cat(preds_f, dim=1)[0]  # [15,C,H,W]
        preds_m = torch.cat(preds_m, dim=1)[0]  # [15,H,W]
        ctx_f, fut_f = context[0][0], future_f[0]  # [5,C,H,W], [15,C,H,W]
        ctx_m, fut_m = context[1][0], future_m[0]  # [5,H,W],   [15,H,W]
    
        # ---- Рисуем КАДРЫ (как раньше) ----
        draw_rollout_grid(ctx_f, fut_f, preds_f, self.writer, self.dir_imgs, iter_, mode='val')
    
        # ---- Рисуем МАСКИ (цветные) ----
        ctx_m_rgb  = masks_to_rgb_tensor(ctx_m)   # [5,3,H,W]
        fut_m_rgb  = masks_to_rgb_tensor(fut_m)   # [15,3,H,W]
        preds_m_rgb= masks_to_rgb_tensor(preds_m) # [15,3,H,W]
        draw_rollout_grid(ctx_m_rgb, fut_m_rgb, preds_m_rgb, self.writer, self.dir_imgs, iter_, mode='val_masks')
    
        # ---- GIF по кадрам ----
        scale = 4
        seq_for_gif = torch.cat([ctx_f, preds_f], dim=0).clamp(0,1)   # [T, C, H, W]
        seq_np = (seq_for_gif * 255).byte().cpu().permute(0,2,3,1).numpy()
        images = []
        for f in seq_np:
            img = Image.fromarray(f)
            if scale != 1.0:
                w, h = img.size
                img = img.resize((int(w*scale), int(h*scale)), Image.NEAREST)
            images.append(img)
        gif_path = os.path.join(self.dir_imgs, f"rollout_val_{iter_:06d}.gif")
        imageio.mimsave(gif_path, images, fps=5)
    
        # ---- GIF по маскам (цветные) ----
        seq_m_np = (torch.cat([ctx_m_rgb, preds_m_rgb], dim=0)
                    .clamp(0,1).mul(255).byte().cpu().permute(0,2,3,1).numpy())
        images_m = []
        for f in seq_m_np:
            img = Image.fromarray(f)
            if scale != 1.0:
                w, h = img.size
                img = img.resize((int(w*scale), int(h*scale)), Image.NEAREST)
            images_m.append(img)
        gif_path_m = os.path.join(self.dir_imgs, f"rollout_val_masks_{iter_:06d}.gif")
        imageio.mimsave(gif_path_m, images_m, fps=5)

    def _log_predictions_next5(self, batch, iter_, step=None):
        frames, masks = batch
        context = (frames[0:1, :5], masks[0:1, :5])  # ([1,5,C,H,W], [1,5,H,W])
        future_f = frames[0:1, 5:10]                 # [1,5,C,H,W]
        future_m = masks[0:1,  5:10]                 # [1,5,H,W]
    
        preds_f, preds_m = [], []
        cur_context = (context[0].clone(), context[1].clone())
    
        self.model.eval()
        with torch.no_grad():
            for t in range(5):
                out_f, out_m, _ = self.model(cur_context)      # ([1,5,C,H,W], [1,5,H,W])
                next_f = out_f[:, -1]                       # [1,C,H,W]
                next_m = out_m[:, -1]                       # [1,H,W]
                preds_f.append(next_f.unsqueeze(1))
                preds_m.append(next_m.unsqueeze(1))
    
                # по-кадровый лог (frames)
                draw_rollout_grid(
                    cur_context[0][0],                      # ctx frames: [5,C,H,W]
                    future_f[0, t, :].unsqueeze(0),        # gt frame:   [1,C,H,W]
                    next_f,                                 # pred frame: [1,C,H,W]
                    self.writer, self.dir_imgs, iter_, mode=f"train_step{t:02d}"
                )
                # по-кадровый лог (masks)
                draw_rollout_grid(
                    masks_to_rgb_tensor(cur_context[1][0]),                 # [5,3,H,W]
                    masks_to_rgb_tensor(future_m[0, t, ...].unsqueeze(0)),  # [1,3,H,W]
                    masks_to_rgb_tensor(next_m),                            # [1,3,H,W]  <-- без squeeze
                    self.writer, self.dir_imgs, iter_, mode=f"train_masks_step{t:02d}"
                )
                    
                # autoregressive update
                cur_context = (
                    torch.cat([cur_context[0][:, 1:], next_f.unsqueeze(1)], dim=1),
                    torch.cat([cur_context[1][:, 1:], next_m.unsqueeze(1)], dim=1),
                )
    
        preds_f = torch.cat(preds_f, dim=1)[0]  # [5,C,H,W]
        preds_m = torch.cat(preds_m, dim=1)[0]  # [5,H,W]
        ctx_f, fut_f = context[0][0], future_f[0]
        ctx_m, fut_m = context[1][0], future_m[0]
    
        # итоговые гриды: frames + masks
        draw_rollout_grid(ctx_f, fut_f, preds_f, self.writer, self.dir_imgs, iter_, mode='train')
        draw_rollout_grid(masks_to_rgb_tensor(ctx_m),
                          masks_to_rgb_tensor(fut_m),
                          masks_to_rgb_tensor(preds_m),
                          self.writer, self.dir_imgs, iter_, mode='train_masks')

        # GIF: frames
        scale = 4
        seq_np = (torch.cat([ctx_f, preds_f], dim=0).clamp(0,1)*255).byte().cpu().permute(0,2,3,1).numpy()
        images = []
        for f in seq_np:
            img = Image.fromarray(f)
            if scale != 1.0:
                w, h = img.size
                img = img.resize((int(w*scale), int(h*scale)), Image.NEAREST)
            images.append(img)
        gif_path = os.path.join(self.dir_imgs, f"rollout_train_{iter_:06d}.gif")
        imageio.mimsave(gif_path, images, fps=5)
    
        # GIF: masks
        seq_m_np = (torch.cat([masks_to_rgb_tensor(ctx_m), masks_to_rgb_tensor(preds_m)], dim=0)
                    .clamp(0,1).mul(255).byte().cpu().permute(0,2,3,1).numpy())
        images_m = []
        for f in seq_m_np:
            img = Image.fromarray(f)
            if scale != 1.0:
                w, h = img.size
                img = img.resize((int(w*scale), int(h*scale)), Image.NEAREST)
            images_m.append(img)
        gif_path_m = os.path.join(self.dir_imgs, f"rollout_train_masks_{iter_:06d}.gif")
        imageio.mimsave(gif_path_m, images_m, fps=5)