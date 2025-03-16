"""
Credit to original code https://github.com/tryolabs/norfair/blob/master/norfair/video.py
Modified to use a user-provided filename suffix for the output file and
to adjust the Video class constructor.
"""

import os
import os.path as osp
import time
from typing import List, Union, Tuple

import cv2
import numpy as np
from rich import print
from rich.progress import BarColumn, Progress, ProgressColumn, TimeRemainingColumn

# Если требуется синхронизация GPU, импортируем torch
try:
    import torch
except ImportError:
    torch = None


def get_terminal_size(default: Tuple[int, int] = (80, 24)) -> Tuple[int, int]:
    columns, lines = default
    for fd in range(0, 3):  # Проверяем стандартные: Stdin, Stdout, Stderr
        try:
            columns, lines = os.get_terminal_size(fd)
        except OSError:
            continue
        break
    return columns, lines


class Video:
    def __init__(self, src: Union[str, int]):
        """
        :param src: Путь к видеофайлу или число (для веб-камеры)
        """
        self.src = src
        is_webcam = lambda x: isinstance(x, int)
        self.display = "webcam" if is_webcam(src) else osp.basename(src)

        # Открываем видеопоток
        self.video_capture = cv2.VideoCapture(src)
        if not self.video_capture.isOpened():
            self._fail(
                f"[bold red]Error:[/bold red] '{self.src}' не является видеофайлом, поддерживаемым OpenCV. "
                "Если видео в порядке, проверьте корректность установки OpenCV."
            )
        self.width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Сохраняем FPS видеопотока; если не удалось получить FPS, ставим запасное значение 25.
        self.fps_capture = self.video_capture.get(cv2.CAP_PROP_FPS) or 25
        self.total_frames = 0 if is_webcam(src) else int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_cnt = 0

        description = f"Run | {self.display}"
        progress_bar_fields: List[Union[str, ProgressColumn]] = [
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            "[yellow]{task.fields[process_fps]:.2f} fps[/yellow]",
        ]
        self.progress_bar = Progress(
            *progress_bar_fields,
            auto_refresh=False,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self.task = self.progress_bar.add_task(
            self.abbreviate_description(description),
            total=self.total_frames,
            start=str(self.src),
            process_fps=0,
        )

    @property
    def fps(self) -> float:
        """Возвращает FPS видео."""
        return self.fps_capture

    def __iter__(self):
        """
        Итератор, возвращающий кадры и временные метки.
        Обновление progress-bar производится каждые 5 кадров для уменьшения накладных расходов.
        Если доступна CUDA, происходит синхронизация GPU каждые 5 кадров.
        """
        with self.progress_bar as progress_bar:
            start_time = time.time()
            timestamp = 0.0
            video_fps = self.fps_capture  # кэшированное значение FPS
            while True:
                ret, frame = self.video_capture.read()
                if not ret or frame is None:
                    break
                self.frame_cnt += 1
                elapsed = time.time() - start_time
                dynamic_fps = self.frame_cnt / elapsed if elapsed > 0 else 0

                # Каждый 5-й кадр обновляем progress-bar, синхронизируем GPU, если необходимо
                if self.frame_cnt % 5 == 0:
                    progress_bar.update(self.task, advance=5, refresh=True, process_fps=dynamic_fps)
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.synchronize()
                else:
                    progress_bar.update(self.task, advance=1, process_fps=dynamic_fps)
                # Принудительная перерисовка progress-bar
                progress_bar.refresh()

                yield frame, timestamp
                timestamp += 1.0 / video_fps
            self.stop()

    def stop(self) -> None:
        self.frame_cnt = 0
        self.video_capture.release()
        cv2.destroyAllWindows()

    def _fail(self, msg: str) -> None:
        raise RuntimeError(msg)

    def show(self, frame: np.ndarray, winname: str = "show", downsample_ratio: float = 1.0) -> int:
        """
        Выводит кадр в указанном окне. При необходимости уменьшает размер кадра.
        """
        if self.frame_cnt == 1:
            cv2.namedWindow(winname, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(winname, 640, 480)
            cv2.moveWindow(winname, 20, 20)
        if downsample_ratio != 1.0:
            frame = cv2.resize(frame, (frame.shape[1] // int(downsample_ratio), frame.shape[0] // int(downsample_ratio)))
        cv2.imshow(winname, frame)
        return cv2.waitKey(1)

    def get_writer(self, frame: np.ndarray, output_path: str, fps: int = 20) -> cv2.VideoWriter:
        """
        Возвращает VideoWriter для записи видео.
        Если output_path не содержит расширение .avi, имя файла формируется на основе исходного видео.
        """
        if not output_path.endswith(".avi"):
            output_path = osp.join(output_path, osp.splitext(osp.basename(self.src))[0] + ".avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        output_size = (frame.shape[1], frame.shape[0])
        writer = cv2.VideoWriter(output_path, fourcc, fps, output_size)
        print(f"[INFO] Writing output to {output_path}")
        return writer

    def get_output_file_path(self, output_folder: str, suffix: List[str] = []) -> str:
        """
        Формирует путь к выходному файлу, используя папку и список суффиксов.
        """
        os.makedirs(output_folder, exist_ok=True)
        base = "webcam" if isinstance(self.src, int) else osp.splitext(self.display)[0]
        filename = f"{base}_{'_'.join(suffix)}" if suffix else base
        output_path = osp.join(output_folder, f"{filename}.avi")
        return output_path

    def abbreviate_description(self, description: str) -> str:
        """
        Сокращает описание, чтобы оно поместилось в ширину терминала.
        """
        terminal_columns, _ = get_terminal_size()
        space_for_description = terminal_columns - 25  # оставляем 25 символов для progress bar
        if len(description) < space_for_description:
            return description
        half = space_for_description // 2 - 3
        return f"{description[:half]} ... {description[-half:]}"


if __name__ == "__main__":
    path = "/home/zmh/hdd/Test_Videos/Tracking/aung_la_fight_cut_1.mp4"
    video = Video(path)
    for frame, ts in video:
        video.show(frame, "debug")
