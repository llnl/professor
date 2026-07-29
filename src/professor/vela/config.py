# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
import logging
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from statistics import mean
from typing import Dict, List, Union, Any
from professor.vela.data import template_config_path
from omegaconf import OmegaConf
from schema import And, Optional, Or, Schema  # ,  SchemaError

logger: logging.Logger = logging.getLogger(__name__)


class Keys(Enum):
    MODEL = "model"
    PYTORCH = "pytorch"
    NAME = "name"
    CHECKPOINTS = "checkpoints"
    DEVICES = "devices"
    SEED = "seed"
    # GPU_0 = "gpu_0"
    # GPU_1 = "gpu_1"
    # GPU_2 = "gpu_2"
    # GPU_3 = "gpu_3"
    # GPU_4 = "gpu_4"
    # GPU_5 = "gpu_5"
    # GPU_6 = "gpu_6"
    # GPU_7 = "gpu_7"
    DIMENSIONS = "dimensions"
    N_INPUTS = "n_inputs"
    BATCH_SIZE = "batch_size"
    Y_PIXELS = "y_pixels"
    X_PIXELS = "x_pixels"
    Z_PIXELS = "z_pixels"
    INVOCATION = "invocation"
    TARGET = "target"
    PARAMS = "params"
    EXECUTION = "execution"
    HALF_PRECISION = "half_precision"
    JIT = "jit"

    GUI = "gui"
    NAPARI = "napari"
    FIELDS = "fields"
    SLIDERS = "sliders"
    LOWER_BOUND = "lower_bound"
    UPPER_BOUND = "upper_bound"
    INITIAL_VALUE = "initial_value"
    MEAN = "mean"
    APPEARANCE = "appearance"
    COLORMAP = "colormap"
    THEME = "theme"
    PLUGINS = "plugins"
    AUTO_LOAD = "auto_load"
    SETTINGS = "settings"
    FLIP_Y = "flip_y"
    REFLECT = "reflect"


# Unfortuantely we are getting a cirular import with vela.utils. This is placed
# here since the Keys class is imported by the utils module.
from professor.vela.utils import get_object_reference  # noqa E402


class VelaSchema(ABC):
    @abstractmethod
    def __init__(self, conf: Union[Dict[str, Any], None] = None) -> None:
        if conf is None:
            self._val = conf
        else:
            self._val = self.schema.validate(conf)

    @property
    @abstractmethod
    def schema(self) -> Schema:
        pass

    @property
    def val(self) -> Union[Dict[str, Any], None]:
        return self._val


# class _ValidateCheckpointSchema(Schema):
#     """
#     Inherits the schema.Schema class for creating a custom validate function which
#     validates that at least one gpu path was set.

#     Note: Have to pass the **kwargs through in validate because other custom
#     validation schemas will have their own key word that is defined.

#     Schema documentation for this: https://github.com/keleshev/schema#customized-validation
#     """

#     def validate(self, data, _is_validate_checkpoint_schema=True, **kwargs):
#         data = super(_ValidateCheckpointSchema, self).validate(
#             data, _is_validate_checkpoint_schema=False, **kwargs
#         )
#         if _is_validate_checkpoint_schema and not self.at_least_one_path(data):
#             raise SchemaError("At least one model path needs to be specified")
#         return data

#     def at_least_one_path(self, data) -> bool:
#         """
#         Check that at least one path is included in the config

#         :param data dict: The config
#         :return: bool
#         """
#         for i in data:
#             print(i)
#         base = "gpu_"
#         for i in range(0, 8):
#             curr = base + str(i)
#             if data.get(curr, None) is not None:
#                 return True
#         return False


# # TODO: Allow to have a list of paths for a GPU
# class CheckpointsSchema(VelaSchema):
#     def __init__(self, conf: Union[Dict, None] = None) -> None:
#         super().__init__(conf)

#     @property
#     def schema(self) -> Schema:
#         return _ValidateCheckpointSchema(
#             {
#                 Optional(Keys.GPU_0.value, default=None): str,
#                 Optional(Keys.GPU_1.value, default=None): str,
#                 Optional(Keys.GPU_2.value, default=None): str,
#                 Optional(Keys.GPU_3.value, default=None): str,
#                 Optional(Keys.GPU_4.value, default=None): str,
#                 Optional(Keys.GPU_5.value, default=None): str,
#                 Optional(Keys.GPU_6.value, default=None): str,
#                 Optional(Keys.GPU_7.value, default=None): str,
#             }
#         )


class DimensionsSchema(VelaSchema):
    def __init__(self, conf: Union[Dict[str, Any], None] = None) -> None:
        super().__init__(conf)

    @property
    def schema(self) -> Schema:
        return Schema(
            {
                Keys.N_INPUTS.value: int,
                Keys.BATCH_SIZE.value: int,
                Keys.Y_PIXELS.value: int,
                Keys.X_PIXELS.value: int,
                Optional(Keys.Z_PIXELS.value, default=1): int,
            }
        )


class _InvocationSchema(Schema):
    """
    Inherits the schema.Schema class for creating a custom validate function which
    populates the args dict with defaults for professor if they are not provided.

    Note: Have to pass the **kwargs through in validate because other custom
    validation schemas will have their own key word that is defined.

    Schema documentation for this: https://github.com/keleshev/schema#customized-validation
    """

    def validate(
        self, data: Dict[str, Any], _is_invocation_schema: bool = True, **kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        data = super(_InvocationSchema, self).validate(data, _is_invocation_schema=False, **kwargs)  # type: ignore
        if _is_invocation_schema:
            return self.professor_defaults(data)
        return data

    def professor_defaults(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.valid_professor_target(data[Keys.TARGET.value]):
            return data

        # It's a valid professor argument target but there are no args -> :(
        # This is most likely an error but out of scope to throw error here
        if data.get(Keys.PARAMS.value, None) is None:
            return data

        if "first_layer" not in data[Keys.PARAMS.value]:
            data[Keys.PARAMS.value]["first_layer"] = {Keys.INVOCATION.value: {Keys.TARGET.value: "torch.nn.identity"}}

        if "last_layer" not in data[Keys.PARAMS.value]:
            data[Keys.PARAMS.value]["last_layer"] = {
                Keys.INVOCATION.value: {
                    Keys.TARGET.value: "professor.layers.AlphaLinear",
                    Keys.PARAMS.value: {"n_channels": data[Keys.PARAMS.value]["num_channels"], "n_dims": 2},
                }
            }

        return data

    def valid_professor_target(self, name: str) -> bool:
        """
        Inspects the signature of the passed professor method and validates
        whether or not it is one that has a first and last layer argument.

        :param name: Name of the professor torch_models method
        :return: Boolean for if it has a first and last layer argument
        """
        if "professor.torch_model" not in name:
            return False

        func = get_object_reference(name)
        argspec = inspect.getfullargspec(func)

        if "first_layer" in argspec.args and "last_layer" in argspec.args:
            return True

        return False


class InvocationSchema(VelaSchema):
    def __init__(self, conf: Union[Dict[str, Any], None] = None) -> None:
        super().__init__(conf)

    @property
    def schema(self) -> Schema:
        return _InvocationSchema(
            # fmt: off
            {Keys.TARGET.value: str, Optional(Keys.PARAMS.value, default=None): dict}
            # fmt: on
        )


class ExecutionSchema(VelaSchema):
    def __init__(self, conf: Union[Dict[str, Any], None] = None) -> None:
        super().__init__(conf)

    @property
    def schema(self) -> Schema:
        return Schema(
            # fmt: off
            {
                Optional(Keys.HALF_PRECISION.value, default=True): bool,
                Optional(Keys.JIT.value, default=True): bool,
            }
            # fmt: on
        )


class ModelSchema(VelaSchema):
    def __init__(self, conf: Union[Dict[str, Any], None] = None) -> None:
        super().__init__(conf)

    @property
    def schema(self) -> Schema:
        # fmt: off
        return Schema({
            Keys.NAME.value: str,
            # Keys.CHECKPOINTS.value: CheckpointsSchema().schema,
            Keys.CHECKPOINTS.value: Schema([str],),
            Keys.DEVICES.value: Schema([int],),
            Keys.DIMENSIONS.value: DimensionsSchema().schema,
            Keys.INVOCATION.value: InvocationSchema().schema,
            Optional(Keys.SEED.value, default=1231): int,
            Optional(
                Keys.EXECUTION.value,
                default=lambda: ExecutionSchema({}).val
            ): ExecutionSchema().schema,
        })

    # fmt: on


class FieldsSchema(VelaSchema):
    def __init__(self, conf: Union[Dict[str, Any], None] = None) -> None:
        super().__init__(conf)

    @property
    def schema(self) -> Schema:
        return Schema([str])


class SlidersSchema(VelaSchema):
    def __init__(self, conf: Union[Dict[str, Any], None] = None) -> None:
        super().__init__(conf)

    @property
    def schema(self) -> Schema:
        # fmt: off
        return Schema(
            {
                str: {
                    Keys.LOWER_BOUND.value: Or(int, float),
                    Keys.UPPER_BOUND.value: Or(int, float),
                    Optional(Keys.INITIAL_VALUE.value, default=Keys.MEAN.value): Or(
                        int, float, Keys.MEAN.value  # type: ignore
                    ),
                }
            }
        )
        # fmt: on


class AppearanceSchema(VelaSchema):
    DEFAULT_COLORMAP = "magma"
    DEFAULT_THEME = "light"
    THEME_OPTIONS = ["light", "dark", "system"]

    def __init__(self, conf: Union[Dict[str, Any], None] = None) -> None:
        super().__init__(conf)

    @property
    def schema(self) -> Schema:
        # fmt: off
        return Schema(
            {
                Optional(Keys.COLORMAP.value, default=self.DEFAULT_COLORMAP): str,
                Optional(Keys.THEME.value, default=self.DEFAULT_THEME): And(
                    str,
                    lambda t: t in ["light", "dark", "system"]
                )
            }
        )
        # fmt: on


class PluginsSchema(VelaSchema):
    def __init__(self, conf: Union[Dict[str, Any], None] = None) -> None:
        super().__init__(conf)

    @property
    def schema(self) -> Schema:
        # fmt: off
        return Schema(
            {
                str: Schema(
                    {
                        Optional(Keys.AUTO_LOAD.value, default=False): bool,
                        object: object
                    },
                )
            }
        )
        # fmt: on


class BoolSchema(VelaSchema):
    def __init__(self, conf: Union[Dict[str, Any], None] = None) -> None:
        super().__init__(conf)

    @property
    def schema(self) -> Schema:
        return Schema(bool)


class GUISchema(VelaSchema):
    def __init__(self, conf: Union[Dict[str, Any], None] = None) -> None:
        super().__init__(conf)

    @property
    def schema(self) -> Schema:
        # fmt: off
        return Schema(
            {
                Keys.FIELDS.value: FieldsSchema().schema,
                Keys.SLIDERS.value: SlidersSchema().schema,
                Optional(
                    Keys.APPEARANCE.value,
                    default=lambda: AppearanceSchema({}).val
                ): AppearanceSchema().schema,
                Optional(Keys.PLUGINS.value, default=None): PluginsSchema().schema,
                Optional(Keys.FLIP_Y.value, default=False): BoolSchema().schema,
                Optional(Keys.REFLECT.value, default=False): BoolSchema().schema,
            }
        )
        # fmt: on


class FullSchema(VelaSchema):
    MODEL_KEYS = [Keys.PYTORCH.value]
    GUI_KEYS = [Keys.NAPARI.value]

    def __init__(self, conf: Union[Dict[str, Any], None] = None) -> None:
        super().__init__(conf)

    @property
    def schema(self) -> Schema:
        # fmt: off
        return Schema(
            {
                Keys.MODEL.value: {Keys.PYTORCH.value: ModelSchema().schema, },
                Keys.GUI.value: {Keys.NAPARI.value: GUISchema().schema, }
            }
        )
        # fmt: on


class Config:
    def __init__(self, cfgpath: str) -> None:
        conf = self._load_cfg(cfgpath)
        self.conf = conf
        dict_conf = OmegaConf.to_container(conf, resolve=True)
        self.dict_conf = dict_conf
        logger.debug(f"Loading config from path: {cfgpath}")

        full = FullSchema(dict_conf).val  # type: ignore
        if full is None:
            raise Exception("An error occured during schema processing")
        self.full: Dict[str, Any] = full

        logger.debug(f"Full config: {self.full}")

        self.model_type: str = self.get_top_level_key_name(model=True)
        self.model: Dict[str, Any] = self.full[Keys.MODEL.value][self.model_type]
        logger.debug(f"{self.model_type}: {self.model}")

        self.model_name: str = self.model[Keys.NAME.value]
        logger.debug(f"{Keys.NAME.value}: {self.model_name}")

        self.checkpoints: List[str] = self.model[Keys.CHECKPOINTS.value]
        logger.debug(f"{Keys.CHECKPOINTS.value}: {self.checkpoints}")

        self.devices: List[str] = self.model[Keys.DEVICES.value]
        logger.debug(f"{Keys.DEVICES.value}: {self.devices}")
        n_dev = len(self.devices)
        n_check = len(self.checkpoints)

        if n_dev != n_check:
            raise ValueError(
                f"You have {n_dev} devices and {n_check}"
                f" checkpoints. These should have the same length"
                f" devices {self.devices} \n"
                f" checkpoints {self.checkpoints}"
            )

        # self.gpu_0 = self.checkpoints[Keys.GPU_0.value]
        # logger.debug(f"{Keys.GPU_0.value}: {self.gpu_0}")

        # self.gpu_1 = self.checkpoints[Keys.GPU_1.value]
        # logger.debug(f"{Keys.GPU_1.value}: {self.gpu_1}")

        # self.gpu_2 = self.checkpoints[Keys.GPU_2.value]
        # logger.debug(f"{Keys.GPU_2.value}: {self.gpu_2}")

        # self.gpu_3 = self.checkpoints[Keys.GPU_3.value]
        # logger.debug(f"{Keys.GPU_3.value}: {self.gpu_3}")

        # self.gpu_4 = self.checkpoints[Keys.GPU_4.value]
        # logger.debug(f"{Keys.GPU_4.value}: {self.gpu_4}")

        # self.gpu_5 = self.checkpoints[Keys.GPU_5.value]
        # logger.debug(f"{Keys.GPU_5.value}: {self.gpu_5}")

        # self.gpu_6 = self.checkpoints[Keys.GPU_6.value]
        # logger.debug(f"{Keys.GPU_6.value}: {self.gpu_6}")

        # self.gpu_7 = self.checkpoints[Keys.GPU_7.value]
        # logger.debug(f"{Keys.GPU_7.value}: {self.gpu_7}")

        # legacy, this sb sefl.devices!
        self.loaded_checkpoints: list[str] = self.devices

        self.seed: int = self.model[Keys.SEED.value]
        logger.debug(f"{Keys.SEED.value}: {self.seed}")

        self.dimensions: Dict[str, int] = self.model[Keys.DIMENSIONS.value]
        logger.debug(f"{Keys.DIMENSIONS.value}: {self.dimensions}")

        self.n_inputs: int = self.dimensions[Keys.N_INPUTS.value]
        logger.debug(f"{Keys.N_INPUTS.value}: {self.n_inputs}")

        self.batch_size: int = self.dimensions[Keys.BATCH_SIZE.value]
        logger.debug(f"{Keys.BATCH_SIZE.value}: {self.batch_size}")

        self.y_pixels: int = self.dimensions[Keys.Y_PIXELS.value]
        logger.debug(f"{Keys.Y_PIXELS.value}: {self.y_pixels}")

        self.x_pixels: int = self.dimensions[Keys.X_PIXELS.value]
        logger.debug(f"{Keys.X_PIXELS.value}: {self.x_pixels}")

        self.z_pixels: int = self.dimensions[Keys.Z_PIXELS.value]
        logger.debug(f"{Keys.Z_PIXELS.value}: {self.z_pixels}")

        self.invocation: Dict[str, Any] = self.model[Keys.INVOCATION.value]
        logger.debug(f"{Keys.INVOCATION.value}: {self.invocation}")

        self.invocation_target: str = self.invocation[Keys.TARGET.value]
        logger.debug(f"Invocation {Keys.TARGET.value}: {self.invocation_target}")

        self.invocation_params: Dict[str, Any] = self.invocation[Keys.PARAMS.value]
        logger.debug(f"Invocation {Keys.PARAMS.value}: {self.invocation_params}")

        self.execution: Dict[str, bool] = self.model[Keys.EXECUTION.value]
        logger.debug(f"{Keys.EXECUTION.value}: {self.execution}")

        self.half_precision: bool = self.execution[Keys.HALF_PRECISION.value]
        logger.debug(f"{Keys.HALF_PRECISION.value}: {self.half_precision}")

        self.jit: bool = self.execution[Keys.JIT.value]
        logger.debug(f"{Keys.JIT.value}: {self.jit}")

        self.gui_type: str = self.get_top_level_key_name(gui=True)
        self.gui: Dict[str, Any] = self.full[Keys.GUI.value][self.gui_type]
        logger.debug(f"{self.gui_type}: {self.model}")

        self.fields: List[str] = self.gui[Keys.FIELDS.value]
        logger.debug(f"{Keys.FIELDS.value}: {self.fields}")

        self.field_map: Dict[str, int] = self.get_field_map()

        self.original_sliders: Dict[str, Dict[str, Any]] = self.gui[Keys.SLIDERS.value]
        logger.debug(f"Original {Keys.SLIDERS.value}: {self.original_sliders}")

        self.sliders: Dict[str, Dict[str, Any]] = self.evaluate_sliders()
        logger.debug(f"Evaluated {Keys.SLIDERS.value}: {self.sliders}")

        self.slider_initial_values: List[float] = self.get_slider_initial_values()

        self.appearance: Dict[str, str] = self.gui[Keys.APPEARANCE.value]
        logger.debug(f"{Keys.APPEARANCE.value}: {self.appearance}")

        self.plugins: Dict[str, Any] = self.gui[Keys.PLUGINS.value]
        logger.debug(f"{Keys.PLUGINS.value}: {self.plugins}")

        self.flip_y: bool = self.gui[Keys.FLIP_Y.value]
        logger.debug(f"{Keys.FLIP_Y.value}: {self.flip_y}")

        self.reflect: bool = self.gui[Keys.REFLECT.value]
        logger.debug(f"{Keys.REFLECT.value}: {self.reflect}")

    def _load_cfg(self, cfgpath: str):
        return OmegaConf.load(cfgpath)

    def get_top_level_key_name(self, model: bool = False, gui: bool = False) -> str:
        """
        Get a top level key name of the model or gui in the config file

        :param model: Return the corresponding Model key
        :param gui: Return the corresponding GUI key
        :return: The string corresponding to the selected key
        :raises ValueError: Can only select either model or gui
        """

        if (not model and not gui) or (model and gui):
            raise ValueError("Either model or gui must be selected.")

        possibilities, search = FullSchema.MODEL_KEYS, Keys.MODEL.value

        if gui:
            possibilities = FullSchema.GUI_KEYS
            search = Keys.GUI.value

        for key in possibilities:
            if key in self.full[search]:
                return key

        return ""  # This case should never be reached since the schema is validated

    # def get_loaded_checkpoints(self) -> :
    #     """
    #     Return a list of the "gpu" class members which were specified in the checkpoints

    #     :return: e.g. {"0": self.gpu_0, "7": self.gpu_7}
    #     """
    #     res = {}

    #     # We are expecting at least one path that has been specified in the config
    #     base = "gpu_"
    #     for i in range(0, 8):

    #         curr = base + str(i)
    #         if self.__dict__[curr] is not None:
    #             res[i] = self.__dict__[curr]

    #     return res

    def multiple_checkpoints_exist(self) -> bool:
        """
        Check if multiple checkpoint paths were provided in the config file

        :return: True or False
        """
        if len(self.loaded_checkpoints) > 1:
            return True
        return False

    def get_field_map(self) -> Dict[str, int]:
        """
        Returns a map of field names to an index value

        :return: e.g. {"materials": 0}
        """
        res = {}
        for i in range(len(self.fields)):
            res[self.fields[i]] = i
        return res

    def evaluate_sliders(self) -> Dict[str, Dict[str, Any]]:
        """
        Converts all "mean" occurences in the sliders dict to the appropriate value

        :return: {"a": {"initial_value": "mean"}} to {"a": {"initial_value": 5.4}}
        """
        res = deepcopy(self.original_sliders)
        for k, v in res.items():
            val = v["initial_value"]
            if val == "mean":
                val = mean((v["lower_bound"], v["upper_bound"]))
                res[k]["initial_value"] = val
        return res

    def get_slider_initial_values(self) -> List[float]:
        """
        Gets the initial value of the slider dictionaries and returns them in
        a list in the order they are listed in the config file

        :return: e.g. [0, 1.2, 3]
        """
        return [val[Keys.INITIAL_VALUE.value] for val in self.sliders.values()]

    # def get_device_type_from_string(self, device: str) -> str:
    #     """
    #     Get the device number from the passed in device string.

    #     :param device: [TODO:description]
    #     :return: [TODO:description]
    #     """
    #     last_char = device[-1]

    #     if not last_char.isdigit():
    #         raise ValueError("Device should be in a form like 'gpu_0'")

    #     return last_char


def build_template_config(
    config_fname: str = "./template_config.yaml",
    model_name: str = "Professor Model",
    checkpoint_path: str = "0000.pt",
    n_inputs: int = 1,
    n_channels: int = 1,
    x_pixels: int = 512,
    y_pixels: int = 512,
    z_pixels: int = 1,
    min_features: int = 128,
    max_features: int = 1024,
    x_kernel: int = 4,
    y_kernel: int = 4,
    z_kernel: int = 0,
    generator_type: str = "legacy",
    act_fun: str = "ReLU",
    upscale_type: str = "transpose",
    fields: list[str] = ["field"],
    input_parameters: list[tuple[str, float, float]] = [("parameter_0", -1.0, 1.0)],
) -> None:
    """
    Build a template visualization config file
    """
    conf: dict[str, Any] = OmegaConf.load(template_config_path)  # type: ignore
    conf["model"]["pytorch"]["name"] = model_name
    conf["model"]["pytorch"]["checkpoints"][0] = os.path.abspath(checkpoint_path)

    dims = conf["model"]["pytorch"]["dimensions"]
    dims["n_inputs"] = n_inputs
    dims["x_pixels"] = x_pixels
    dims["y_pixels"] = y_pixels
    if z_pixels:
        dims["z_pixels"] = z_pixels

    available_generators = {
        "legacy": "professor.torch_models.Generator",
        "2D": "professor.torch_models.Generator2D",
        "3D-triplane": "professor.torch_models.Generator3DTriplane",
        "3D-spectral": "professor.torch_models.Generator3DSpectral",
        "3D-voxel": "professor.torch_models.Generator3DVoxel",
    }
    conf["model"]["pytorch"]["invocation"]["target"] = available_generators[generator_type]

    if "3D" in generator_type:
        conf["model"]["pytorch"]["invocation"]["params"]["last_layer"]["invocation"]["params"]["n_dims"] = 3

    # Note: some layers in this model type do not support half precision
    if generator_type == "3D-spectral":
        conf["model"]["pytorch"]["execution"]["half_precision"] = False

    params = conf["model"]["pytorch"]["invocation"]["params"]
    params["num_channels"] = n_channels
    params["min_features"] = min_features
    params["max_features"] = max_features
    params["x_kernel"] = x_kernel
    params["y_kernel"] = y_kernel
    params["act_fun"] = act_fun

    if z_kernel:
        params["z_kernel"] = z_kernel

    if generator_type in ["2D", "3D-triplane", "3D-spectral", "3D-voxel"]:
        params["upscale_type"] = upscale_type

    conf["gui"]["napari"]["fields"] = fields

    sliders = conf["gui"]["napari"]["sliders"]
    for param_name, param_min, param_max in input_parameters:
        sliders[param_name] = {
            "lower_bound": param_min,
            "upper_bound": param_max,
            "initial_value": 0.5 * (param_min + param_max),
        }

    config_fname = os.path.abspath(os.path.expanduser(config_fname))
    config_root = os.path.dirname(config_fname)
    os.makedirs(config_root, exist_ok=True)
    OmegaConf.save(config=conf, f=config_fname)


def update_config_checkpoint(config_fname: str = "./template_config.yaml", checkpoint_path: str = "0000.pt") -> None:
    """
    Update the checkpoint path value in a configuration file
    """
    conf: dict[str, Any] = OmegaConf.load(config_fname)  # type: ignore
    conf["model"]["pytorch"]["checkpoints"][0] = os.path.abspath(checkpoint_path)
    OmegaConf.save(config=conf, f=config_fname)
