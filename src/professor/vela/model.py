# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import socket
from collections import OrderedDict
from typing import Any, List, Optional
import numpy as np
import torch
from torchvision.transforms.functional import vflip
from professor.vela.config import Config
from professor.vela.utils import (
    get_object_reference,
    initialize_function_calls_in_parameters,
)
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from napari.layers import Image

logger: logging.Logger = logging.getLogger(__name__)


class ModelCollection:
    """
    Object for storing a generic model. Type hints should be added when new
    model types are implemented.

    `inference`: Holds the reference to the model object, and should be used
    to calculate a new inference.

    `image`: Holds the most recently generated image for the model
    """

    def __init__(
        self,
        model: torch.nn.Module,
        image: torch.Tensor,
        tensor: torch.Tensor,
        path: str,
        device: torch.device,
        name: str,
        flip_y: bool,
        reflect: bool,
    ) -> None:
        self.ref: torch.nn.Module = model
        self.image: torch.Tensor = image
        self.tensor: torch.Tensor = tensor
        self.path: str = path
        self.device: torch.device = device
        self.name: str = name
        self.gui_layer: Optional[Image] = None
        self.flip_y: bool = flip_y
        self.reflect: bool = reflect

    def inference(self) -> None:
        # NOTE: Everytime model inference, put in torch.no_grad context
        with torch.no_grad():
            if self.reflect:
                image = self.ref(self.tensor)
                if self.flip_y:
                    image = vflip(image)
                self.image[:] = torch.cat((vflip(image[:, :, 1:, :]), image), dim=2)
            else:
                self.image[:] = self.ref(self.tensor)
                if self.flip_y:
                    self.image[:] = vflip(self.image)
        # print('self image shape', self.image.shape)


class Model(ABC):
    MAX_CHECKPOINTS: int = 8
    CHECKPOINT_STR: str = "checkpoint_"

    @abstractmethod
    def __init__(self, config: Config) -> None:
        logger.info(f"Number of GPU's on {socket.gethostname()} is {self.num_gpus}")
        self._config: Config = config
        self.models: List[ModelCollection] = []

    @property
    @abstractmethod
    def num_gpus(self) -> int:
        """
        The number of GPU's available on the host

        :return: int
        """
        pass

    @property
    def num_checkpoints(self) -> int:
        """
        Returns the upper bound of the number of checkpoints that have been set
        :return: int
        """
        return len(self._config.checkpoints)

    @abstractmethod
    def _load_model(self, model: Any, path: str, device: Any) -> Any:
        """
        Loads the model from the checkpoint path onto the specified device
        and returns a reference to it's evaluate method. When it's loaded,
        print_summary() should also be invoked.

        :param model_ref: The result of calling _get_model_ref()
        :param path: The path of the weight file to be loaded
        :param device: The device that the model should be loaded onto
        :return: A model object that is ready to be evaluated
        """
        pass

    @abstractmethod
    def _get_tensor(self, device: Any) -> Any:
        """
        Initializes the initial tensor for the model and returns a reference
        to it.
        """
        pass

    @abstractmethod
    def _move_tensor_to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Moves the tensor to cpu and pins the memory if possible.
        """

    def _load_checkpoints(self) -> None:
        """
        Load the checkpoints onto their respective devices and set the appropriate
        properties of the ModelCollection.
        """
        checkpoint_index = 0
        for device_idx, path in enumerate(self._config.checkpoints):
            if path is not None and checkpoint_index < self.num_checkpoints:
                device_id = self._config.devices[device_idx]
                device = self._get_checkpoint_device(device_id)
                model = self._load_model(self._get_model_ref(), path, device)
                tensor = self._get_tensor(device)
                with torch.no_grad():
                    if self._config.reflect:
                        image = model(tensor)
                        if self._config.flip_y:
                            image = vflip(image)
                        image = torch.cat((vflip(image[:, :, 1:, :]), image), dim=2)
                    else:
                        image = model(tensor)
                        if self._config.flip_y:
                            image = vflip(image)
                image = self._move_tensor_to_cpu(image)
                file_name = path.rsplit("/", 1)[-1]
                name = f"{file_name} | {device}"

                model_object = ModelCollection(
                    model,
                    image,
                    tensor,
                    path,
                    device,
                    name,
                    self._config.flip_y,
                    self._config.reflect,
                )

                self.models.append(model_object)

                # # support static analysis
                # if checkpoint_index == 0:
                #     self.checkpoint_0 = model_object
                # elif checkpoint_index == 1:
                #     self.checkpoint_1 = model_object
                # elif checkpoint_index == 2:
                #     self.checkpoint_2 = model_object
                # elif checkpoint_index == 3:
                #     self.checkpoint_3 = model_object
                # elif checkpoint_index == 4:
                #     self.checkpoint_4 = model_object
                # elif checkpoint_index == 5:
                #     self.checkpoint_5 = model_object
                # elif checkpoint_index == 6:
                #     self.checkpoint_6 = model_object
                # elif checkpoint_index == 7:
                #     self.checkpoint_7 = model_object

                logger.debug(f"Loaded checkpoint_{checkpoint_index}")
                checkpoint_index += 1

    @abstractmethod
    def _get_checkpoint_device(self, device_id: str) -> torch.device:
        """
        Return an appropriate device based on the device_id string passed. The
        id should be just a singular num.

        :param device_id: ID string for the device e.g. "0", "2"
        :return: The set device
        """
        pass

    def _get_model_ref(self) -> Any:
        params = initialize_function_calls_in_parameters(self._config.invocation_params)
        model_target = get_object_reference(self._config.invocation_target)
        return model_target(**params)

    def get_pixelwise_mean_and_sd(self, channel: int = 0) -> Any:
        """
        Returns the pixelwise mean of all of the checkpoint images

        :return: [TODO:description]
        """
        imgs = []
        for i in range(0, self.num_checkpoints):
            # c = self.CHECKPOINT_STR + str(i)
            current_checkpoint = self.models[i]
            imgs.append(current_checkpoint.image[0, channel].cpu().numpy())

        mean = np.mean(imgs, axis=0)
        std = np.std(imgs, axis=0, ddof=1)

        return mean, std


class PyTorchModel(Model):
    """
    Instantiate a torch model (.pt). Supports normal CUDA and MPS devices
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._set_seed()

        self._load_checkpoints()

        for i in range(0, self.num_checkpoints):
            # c = self.CHECKPOINT_STR + str(i)
            # current_checkpoint = self.__dict__[c]

            self._set_half_precision(self.models[i])
            self._set_jit_tracing(self.models[i])

    @property
    def num_gpus(self) -> int:
        return torch.cuda.device_count()

    def _get_checkpoint_device(self, device_id: str) -> torch.device:
        if torch.cuda.is_available():
            logger.info(f"CUDA is available. Setting cuda:{device_id}")
            return torch.device(f"cuda:{device_id}")
        elif torch.backends.mps.is_available():
            logger.info("MPS is available. Using one device.")
            return torch.device("mps")
        else:
            logging.warning(
                " CUDA or MPS not available. Setting torch device to CPU. " + "model inference will be slow."
            )
            # d = torch.device()
            return torch.device("cpu")

    def _load_model(self, model: Any, path: str, device: torch.device) -> None:
        # NOTE: There is a bug reported specifically for the RadioConvPixelShuffel model
        # where you have to make a forward call in the cpu before being able
        # to move to the gpu. This doesn't seem to break the other models.
        X_val = torch.zeros(1, self._config.n_inputs, 1, 1).to("cpu")
        model(X_val)

        model.to(device)
        if device.type == "cuda":
            map_loc = {"cuda:0": f"{device.type}:{device.index}"}
        else:
            map_loc = {"cuda:0": f"{device.type}"}
        checkpoint = torch.load(path, map_location=map_loc)

        logger.debug(f"Checkpoint keys: {checkpoint.keys()}")

        new_state_dict = OrderedDict()
        for k, v in checkpoint["model"].items():
            name = k.replace("module.", "", 1)
            new_state_dict[name] = v
            logger.debug(f"Key Before: {k}")
            logger.debug(f"Key After: {name}")

        model.load_state_dict(new_state_dict, strict=True)
        model = model.eval()

        return model

    def _get_tensor(self, device: torch.device):
        tensor = torch.tensor(self._config.slider_initial_values).float()
        return tensor.view(1, -1, 1, 1).to(device)

    def _set_seed(self):
        np.random.seed(self._config.seed)
        torch.manual_seed(self._config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self._config.seed)

    def _set_half_precision(self, current_model: ModelCollection):
        if self._config.half_precision:
            logger.info(f"Enabling half precision for model on {current_model.device}")
            current_model.ref = current_model.ref.half()
            current_model.tensor = current_model.tensor.half()

    def _set_jit_tracing(self, current_model: ModelCollection):
        if self._config.jit:
            logger.info(f"Initializing JIT tracing for model on {current_model.device}")
            current_model.ref = torch.jit.trace(current_model.ref, current_model.tensor)  # type: ignore
            # do we still need to freeze in modern pytorch?
            # current_model.ref = torch.jit.freeze(current_model.ref)
            with torch.no_grad():
                _ = current_model.ref(current_model.tensor)
                _ = current_model.ref(current_model.tensor)
        else:
            logger.info(f"Jit tracing will not be performe for model on {current_model.device}")
            with torch.no_grad():
                _ = current_model.ref(current_model.tensor)

    def _move_tensor_to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        t = tensor.detach().cpu()
        # NOTE: Memory pinning is not working
        if torch.cuda.is_available():
            t.pin_memory()
        return t
