import math
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class Ploter:
    """
    Утилита для визуализации логов обучения из CSV со столбцами вроде:
    epoch | loss | mean_loss | Loss | SSIM | PSNR | LPIPS | FID | FVD

    - "loss" (маленькая 'l') трактуется как тренировочный лосс по итерациям.
    - "Loss" (большая 'L') трактуется как валидационный лосс (обычно раз в N итераций).
    - Если у вас альтернативные названия (loss_iter / loss_epoch), класс попытается их распознать.
    """

    def __init__(self, csv_path: Union[str, Path]):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        # очистим заголовки
        self.df.columns = [str(c).strip() for c in self.df.columns]
        # нормализуем известные имена
        self.df = self._normalize_columns(self.df)
        # приводим типы
        for c in self.df.columns:
            if c.lower() in {"epoch", "loss", "loss_iter", "mean_loss", "loss_epoch", "ssim", "psnr", "lpips", "fid", "fvd"}:
                self.df[c] = pd.to_numeric(self.df[c], errors="coerce")
        # если нет epoch — создадим
        if "epoch" not in self.df.columns:
            self.df["epoch"] = 0
        # общий индекс итераций
        self.df = self.df.reset_index(drop=False).rename(columns={"index": "iteration"})
        # шаг внутри эпохи (0-based)
        self.df["step_in_epoch"] = self.df.groupby("epoch").cumcount()

    # ===================== ПУБЛИЧНОЕ API =====================

    def plot_train_val_loss(
        self,
        logy: bool = False,
        smooth: Optional[str] = None,      # None | 'ema' | 'ma'
        ema_alpha: float = 0.1,
        ma_window: int = 101,
        epoch_base: int = 1,
        epoch_tick_step: int = 1,
        max_epochs: Optional[int] = None,
        hide_first_epoch_label: bool = True,
        savepath: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (9, 5),
        title: Optional[str] = "Training vs Validation Loss",
    ) -> None:
        """
        Train vs Validation (валидация берётся из столбца 'Loss'/'loss_epoch', если есть).
        Тики — только старты эпох; подписи — непрерывные 1..N (epoch_base задаёт сдвиг).
        """
        df = self._slice_first_n_epochs(self.df, max_epochs=max_epochs)
    
        train_col = self._get_first_existing(["loss_iter", "loss"], df.columns)
        val_col   = self._get_first_existing(["loss_epoch", "Loss", "val_loss"], df.columns)
        if train_col is None:
            raise ValueError("Training loss column not found. Expected one of: loss_iter, loss.")
    
        x = df["iteration"].to_numpy()
        y_train = df[train_col].to_numpy()
    
        # сглаживание train
        y_train_smooth = None
        if smooth == "ema":
            y_train_smooth = self._ema(y_train, alpha=ema_alpha)
        elif smooth == "ma":
            y_train_smooth = self._moving_average(y_train, window=ma_window)
    
        # validation (разреженно)
        x_val = y_val = np.array([])
        if val_col is not None:
            mask_val = df[val_col].notna().to_numpy()
            x_val = x[mask_val]
            y_val = df.loc[mask_val, val_col].to_numpy()
    
        # --- plot ---
        plt.figure(figsize=figsize)
        plt.plot(x, y_train, label="Train Loss")
        if y_train_smooth is not None:
            plt.plot(x[: len(y_train_smooth)], y_train_smooth, label=f"Train Loss (smoothed)")
        if val_col is not None and len(x_val) > 0:
            plt.plot(x_val, y_val, label="Validation Loss")
    
        if logy:
            plt.yscale("log")
    
        ticks_pos, ticks_lbl = self._epoch_ticks(
            df,
            epoch_base=epoch_base,
            step=epoch_tick_step,
            label_mode="continuous",   # непрерывные подписи 1..N
        )
        if hide_first_epoch_label and len(ticks_pos) > 0:
            ticks_pos, ticks_lbl = ticks_pos[1:], ticks_lbl[1:]
        if len(ticks_pos) > 0:
            plt.xticks(ticks_pos, ticks_lbl)
    
        plt.xlabel("Epochs")
        plt.ylabel("loss")
        if title:
            plt.title(title)
        plt.legend()
        if len(x) > 0:
            plt.xlim(left=x.min(), right=x.max())
        plt.tight_layout()
        if savepath is not None:
            savepath = Path(savepath)
            savepath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(savepath, dpi=150, bbox_inches="tight")
        self._save_pdf(plt.gcf(), "train_val_loss.pdf")
        plt.show()

    def plot_val_metrics_combined(
        self,
        metrics: Iterable[str] = ("Loss","SSIM","FID","FVD"),
        epoch_base: int = 1,
        logy: bool = False,
        epoch_tick_step: int = 1,
        max_epochs: Optional[int] = None,
        hide_first_epoch_label: bool = True,
        normalize: Optional[str] = None,      # None | 'zscore' | 'minmax'
        figsize: Tuple[int,int] = (10, 5),
        savepath: Optional[Union[str, Path]] = None,
        title: str = "Validation metrics",
    ) -> None:
        """
        ВСЕ указанные метрики валидации — на одном графике.
        Разные масштабы можно выровнять normalize='zscore' или 'minmax'.
        """
        df_all = self._slice_first_n_epochs(self.df, max_epochs=max_epochs)
        colmap = {name: self._resolve_metric_column(name, df_all.columns) for name in metrics}
        colmap = {name: col for name, col in colmap.items() if col is not None}
        if not colmap:
            raise ValueError("Нет доступных метрик для отрисовки.")
    
        def _norm(vals: np.ndarray, how: Optional[str]):
            if how is None:
                return vals
            if how == "zscore":
                m = np.nanmean(vals); s = np.nanstd(vals)
                return (vals - m) / (s if s > 0 else 1.0)
            if how == "minmax":
                mn = np.nanmin(vals); mx = np.nanmax(vals)
                return (vals - mn) / (mx - mn) if mx > mn else vals
            return vals
    
        plt.figure(figsize=figsize)
        for name, col in colmap.items():
            dfm = df_all.loc[df_all[col].notna(), ["iteration", col]].rename(columns={col: name})
            x = dfm["iteration"].to_numpy()
            y = _norm(dfm[name].to_numpy(), normalize)
            plt.plot(x, y, label=name)
    
        if logy:
            plt.yscale("log")
    
        ticks_pos, ticks_lbl = self._epoch_ticks(
            df_all,
            epoch_base=epoch_base,
            step=epoch_tick_step,
            label_mode="continuous",
        )
        if hide_first_epoch_label and len(ticks_pos) > 0:
            ticks_pos, ticks_lbl = ticks_pos[1:], ticks_lbl[1:]
        if len(ticks_pos) > 0:
            plt.xticks(ticks_pos, ticks_lbl)
    
        plt.xlabel("Epochs")
        plt.ylabel("Value" if normalize is None else f"value ({normalize})")
        plt.title(title)
        plt.legend()
        if len(df_all) > 0:
            plt.xlim(df_all["iteration"].iloc[0], df_all["iteration"].iloc[-1])
        plt.tight_layout()
        if savepath is not None:
            savepath = Path(savepath)
            savepath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(savepath, dpi=150, bbox_inches="tight")
        self._save_pdf(plt.gcf(), "val_metrics_combined.pdf")
        plt.show()

    def plot_val_metrics(
        self,
        metrics: Iterable[str] = ("Loss", "SSIM", "FID", "FVD"),
        epoch_base: int = 1,
        logy_for: Iterable[str] = (),
        epoch_tick_step: int = 1,
        max_epochs: Optional[int] = None,
        epoch_tick_position: str = "start",   # 'start' | 'end' | 'both'
        add_last_epoch_end_tick: bool = False,
        tick_dedup_prefer: str = "end",
        figsize: Tuple[int, int] = (9, 5),
        saveprefix: Optional[Union[str, Path]] = None,
        title_prefix: str = "Validation ",
    ) -> List[Path]:
        """
        Рисует отдельный график для каждой метрики валидации.
        Возвращает пути к сохранённым файлам, если указан saveprefix.
        """
        saved: List[Path] = []
        df_all = self._slice_first_n_epochs(self.df, max_epochs=max_epochs)

        colmap = {name: self._resolve_metric_column(name, df_all.columns) for name in metrics}

        for name, col in colmap.items():
            if col is None:
                continue
            df = df_all.copy()
            mask = df[col].notna().to_numpy()
            df = df.loc[mask]
            x = df["iteration"].to_numpy()
            y = df[col].to_numpy()

            plt.figure(figsize=figsize)
            plt.plot(x, y, label=name)
            if name in set(logy_for):
                plt.yscale("log")

            ticks_pos, ticks_lbl = self._epoch_ticks(
                df,
                epoch_base=epoch_base,
                step=epoch_tick_step,
                position=epoch_tick_position,
                add_last_end=add_last_epoch_end_tick,
                dedup_prefer=tick_dedup_prefer,
            )
            if len(ticks_pos) > 0:
                plt.xticks(ticks_pos, ticks_lbl)

            plt.xlabel("Epochs")
            plt.ylabel(name.lower())
            plt.title(f"{title_prefix}{name}")
            plt.legend()
            if len(x) > 0:
                plt.xlim(left=x.min(), right=x.max())
            plt.tight_layout()

            if saveprefix is not None:
                savepath = Path(f"{saveprefix}.{name}.png")
                savepath.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(savepath, dpi=150, bbox_inches="tight")
                saved.append(savepath)
            self._save_pdf(plt.gcf(), f"val_{name}.pdf")
            plt.show()

        return saved

    # ===================== ВСПОМОГАТЕЛЬНЫЕ =====================

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        cols = list(df.columns)
        rename = {}
        if "Loss" in cols and "loss_epoch" not in cols:
            rename["Loss"] = "Loss"
        if "FID" not in cols and "fid" in cols:
            rename["fid"] = "FID"
        if "FVD" not in cols and "fvd" in cols:
            rename["fvd"] = "FVD"
        return df.rename(columns=rename)

    @staticmethod
    def _get_first_existing(candidates: Iterable[str], existing: Iterable[str]) -> Optional[str]:
        exist = set(existing)
        for c in candidates:
            if c in exist:
                return c
        return None

    @staticmethod
    def _ema(y: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        out = np.empty_like(y)
        s = 0.0
        initialized = False
        for i, v in enumerate(y):
            if np.isfinite(v):
                if not initialized:
                    s = v
                    initialized = True
                else:
                    s = alpha * v + (1 - alpha) * s
                out[i] = s
            else:
                out[i] = np.nan
        return out

    @staticmethod
    def _moving_average(y: np.ndarray, window: int = 101) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if window < 1:
            return y
        finite = np.isfinite(y).astype(float)
        y0 = np.where(np.isfinite(y), y, 0.0)
        kernel = np.ones(window, dtype=float)
        num = np.convolve(y0, kernel, mode="same")
        den = np.convolve(finite, kernel, mode="same")
        with np.errstate(invalid="ignore"):
            ma = num / den
        return ma

    @staticmethod
    def _epoch_starts(df: pd.DataFrame) -> Tuple[List[int], List[int]]:
        first_pos_mask = df["epoch"].ne(df["epoch"].shift(fill_value=df["epoch"].iloc[0])).to_numpy()
        pos = df.index[first_pos_mask].tolist()
        epochs = df.loc[pos, "epoch"].astype(int).tolist()
        return pos, epochs

    @staticmethod
    def _epoch_ends(df: pd.DataFrame) -> Tuple[List[int], List[int]]:
        # конец эпохи — это строка перед началом следующей; для последней — последний индекс
        starts, epochs = Ploter._epoch_starts(df)
        if not starts:
            return [], []
        ends = [s - 1 for s in starts[1:]] + [df.index[-1]]
        return ends, epochs

    @staticmethod
    def _epoch_ticks(
        df: pd.DataFrame,
        epoch_base: int = 1,
        step: int = 1,
        label_mode: str = "continuous",   # 'continuous' | 'from_csv'
    ) -> Tuple[List[int], List[str]]:
        """
        ТОЛЬКО старты эпох.
        label_mode='continuous' — подписи 1..N (не зависят от нумерации CSV);
        label_mode='from_csv'   — подписи как в CSV + epoch_base.
        """
        # индексы первых строк каждой эпохи (включая самую первую)
        start_mask = df["epoch"].shift(1) != df["epoch"]
        if len(start_mask) > 0:
            start_mask.iloc[0] = True
        pos_all = df.index[start_mask].tolist()
        epochs_csv = df.loc[pos_all, "epoch"].astype(int).tolist()
    
        # подписи
        if label_mode.lower() == "from_csv":
            labels_all = [str(int(e + epoch_base)) for e in epochs_csv]
        else:
            labels_all = [str(epoch_base + i) for i in range(len(pos_all))]
    
        # прореживание
        step = max(1, int(step))
        pos = pos_all[::step]
        labels = labels_all[::step]
    
        # индексы -> iteration
        tick_pos = df.loc[pos, "iteration"].astype(int).tolist()
        return tick_pos, labels

    def _slice_first_n_epochs(self, df: pd.DataFrame, max_epochs: Optional[int]) -> pd.DataFrame:
        """
        Ровно первые N эпох: находим старты эпох и обрезаем по началу (N+1)-й.
        Работает корректно даже если в логе есть перезапуск (эпохи снова с 0).
        """
        if max_epochs is None or max_epochs <= 0:
            return df.copy()
        # индексы первых строк каждой эпохи
        start_mask = df["epoch"].shift(1) != df["epoch"]
        start_mask.iloc[0] = True
        starts = df.index[start_mask].tolist()
        
        if len(starts) <= max_epochs:
            # в логе меньше или ровно N эпох — берём всё до конца
            return df.copy()
        # обрезаем строго до начала (N+1)-й эпохи
        cutoff = starts[max_epochs]
        return df.iloc[:cutoff].copy()

    @staticmethod
    def _resolve_metric_column(name: str, columns: Iterable[str]) -> Optional[str]:
        candidates = [name, name.upper(), name.lower(), name.capitalize()]
        for c in candidates:
            if c in columns:
                return c
        if name.upper() == "LOSS":
            for c in ["Loss", "loss_epoch"]:
                if c in columns:
                    return c
        return None

    def _save_pdf(self, fig, filename: str) -> Path:
        out_dir = self.csv_path.parent / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = out_dir / filename
        with PdfPages(str(pdf_path)) as pdf:
            pdf.savefig(fig)
        return pdf_path