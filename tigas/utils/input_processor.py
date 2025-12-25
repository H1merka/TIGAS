"""
Обработчик входных данных для TIGAS API.
Централизует логику преобразования различных типов входов в тензоры.
"""

import torch
from pathlib import Path
from PIL import Image
from typing import Union, List, Optional
from ..data.transforms import get_inference_transforms


class InputProcessor:
    """
    Обработчик входных данных для TIGAS.
    Преобразует различные форматы входных данных в тензоры PyTorch.
    """

    def __init__(
        self,
        img_size: int = 256,
        device: Optional[str] = None,
        normalize: bool = True
    ):
        """
        Args:
            img_size: Размер изображения
            device: Устройство для вычислений
            normalize: Нормализовать ли изображения
        """
        self.img_size = img_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = get_inference_transforms(
            img_size=img_size,
            normalize=normalize
        )

    def process(
        self,
        x: Union[torch.Tensor, str, Path, Image.Image, List]
    ) -> torch.Tensor:
        """
        Обработать входные данные и преобразовать в тензор.

        Args:
            x: Входные данные (тензор, путь, PIL Image, список)

        Returns:
            Обработанный тензор [B, C, H, W]
        """
        if isinstance(x, (str, Path)):
            return self._process_path(x)
        elif isinstance(x, Image.Image):
            return self._process_pil_image(x)
        elif isinstance(x, list):
            return self._process_list(x)
        elif isinstance(x, torch.Tensor):
            return self._process_tensor(x)
        else:
            raise TypeError(
                f"Неподдерживаемый тип входных данных: {type(x)}. "
                f"Поддерживаются: torch.Tensor, str, Path, PIL.Image, List"
            )

    def _process_path(self, path: Union[str, Path]) -> torch.Tensor:
        """Обработать путь к файлу."""
        path = Path(path)
        if path.is_dir():
            raise ValueError(
                f"Путь указывает на директорию: {path}. "
                f"Используйте compute_directory() для обработки директорий."
            )
        
        img = Image.open(path).convert('RGB')
        return self._process_pil_image(img)

    def _process_pil_image(self, img: Image.Image) -> torch.Tensor:
        """Обработать PIL изображение."""
        tensor = self.transform(img)
        tensor = tensor.unsqueeze(0)  # [1, C, H, W]
        return tensor.to(self.device)

    def _process_list(self, items: List) -> torch.Tensor:
        """Обработать список входных данных."""
        processed = []
        for item in items:
            if isinstance(item, (str, Path)):
                img = Image.open(item).convert('RGB')
                tensor = self.transform(img)
            elif isinstance(item, Image.Image):
                tensor = self.transform(item)
            elif isinstance(item, torch.Tensor):
                tensor = item
            else:
                raise TypeError(
                    f"Неподдерживаемый тип в списке: {type(item)}"
                )
            processed.append(tensor)
        
        stacked = torch.stack(processed)
        return stacked.to(self.device)

    def _process_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Обработать тензор."""
        # Добавить batch dimension если нужно
        if x.ndim == 3:
            x = x.unsqueeze(0)
        
        return x.to(self.device)



