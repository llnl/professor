# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from schema import SchemaError, SchemaMissingKeyError, SchemaWrongKeyError
from professor.vela.config import (
    AppearanceSchema,
    # CheckpointsSchema,
    ExecutionSchema,
    FieldsSchema,
    FullSchema,
    GUISchema,
    Keys,
    DimensionsSchema,
    InvocationSchema,
    ModelSchema,
    PluginsSchema,
    SlidersSchema,
)


# class TestCheckpointSchema:
#     def test_normal_case(self, checkpoints_conf):
#         CheckpointsSchema(checkpoints_conf)

#     def test_one_gpu_set(self, checkpoints_conf):
#         checkpoints_conf.pop(Keys.GPU_0.value)
#         CheckpointsSchema(checkpoints_conf)

#     def test_no_gpu_paths_set(self, checkpoints_conf):
#         checkpoints_conf.pop(Keys.GPU_0.value)
#         checkpoints_conf.pop(Keys.GPU_1.value)
#         with pytest.raises(SchemaError):
#             CheckpointsSchema(checkpoints_conf)


class TestDimensionsSchema:
    def test_normal_case(self, dimensions_conf):
        DimensionsSchema(dimensions_conf)

    def test_missing_keys(self, dimensions_conf):
        for key in dimensions_conf.keys():
            copy = dimensions_conf.copy()
            copy.pop(key, None)
            with pytest.raises(SchemaMissingKeyError):
                DimensionsSchema(copy)

    def test_bad_types(self, dimensions_conf):
        for key in dimensions_conf.keys():
            copy = dimensions_conf.copy()
            copy[key] = "str not allowed"
            with pytest.raises(SchemaError):
                DimensionsSchema(copy)

    def test_val_set(self, dimensions_conf):
        assert DimensionsSchema(dimensions_conf).val == dimensions_conf


class TestInvocationSchema:
    def test_normal_case(self, invocation_conf):
        InvocationSchema(invocation_conf)

    def test_missing_target(self, invocation_conf):
        invocation_conf.pop(Keys.TARGET.value)
        with pytest.raises(SchemaMissingKeyError):
            InvocationSchema(invocation_conf)

    def test_missing_args(self, invocation_conf):
        invocation_conf.pop(Keys.PARAMS.value)
        res = InvocationSchema(invocation_conf).val
        assert res[Keys.PARAMS.value] is None

    def test_bad_invocation_str(self, invocation_conf):
        invocation_conf[Keys.TARGET.value] = 12
        with pytest.raises(SchemaError):
            InvocationSchema(invocation_conf)

    def test_bad_args_dict(self, invocation_conf):
        invocation_conf[Keys.PARAMS.value] = [1, 2, 3]
        with pytest.raises(SchemaError):
            InvocationSchema(invocation_conf)

    def test_bad_professor_target(self, invocation_conf):
        """
        If the target is not a valid professor signature, then the args
        value should be populated for first_layer or last_layer
        """
        invocation_conf[Keys.TARGET.value] = "professor.torch_models.ConvAutoencoder"
        invocation_conf[Keys.PARAMS.value].pop("first_layer")
        invocation_conf[Keys.PARAMS.value].pop("last_layer")
        res = InvocationSchema(invocation_conf).val
        with pytest.raises(KeyError):
            res[Keys.PARAMS.value]["first_layer"]
        with pytest.raises(KeyError):
            res[Keys.PARAMS.value]["last_layer"]

    def test_bad_professor_target_name(self, invocation_conf):
        invocation_conf[Keys.TARGET.value] = "not.valid"
        invocation_conf[Keys.PARAMS.value].pop("first_layer")
        res = InvocationSchema(invocation_conf).val
        with pytest.raises(KeyError):
            res[Keys.PARAMS.value]["first_layer"]


class TestExecutionSchema:
    def test_normal_case(self, execution_conf):
        ExecutionSchema(execution_conf)

    def test_missing_half_precision(self, execution_conf):
        execution_conf.pop(Keys.HALF_PRECISION.value)
        res = ExecutionSchema(execution_conf).val
        assert res == {
            Keys.HALF_PRECISION.value: True,
            Keys.JIT.value: True,
        }

    def test_missing_jit(self, execution_conf):
        execution_conf.pop(Keys.JIT.value)
        res = ExecutionSchema(execution_conf).val
        assert res == {
            Keys.HALF_PRECISION.value: True,
            Keys.JIT.value: True,
        }

    def test_missing_all(self):
        res = ExecutionSchema({}).val
        assert res == {
            Keys.HALF_PRECISION.value: True,
            Keys.JIT.value: True,
        }


class TestModelSchema:
    def test_normal_case(self, pytorch_conf):
        ModelSchema(pytorch_conf)

    def test_missing_name(self, pytorch_conf):
        pytorch_conf.pop(Keys.NAME.value)
        with pytest.raises(SchemaError):
            ModelSchema(pytorch_conf)

    def test_name_type(self, pytorch_conf):
        pytorch_conf[Keys.NAME.value] = 1234
        with pytest.raises(SchemaError):
            ModelSchema(pytorch_conf)

    def test_missing_paths(self, pytorch_conf):
        res = ModelSchema(pytorch_conf).val
        checkpoints = res[Keys.CHECKPOINTS.value]  # noqa F841
        # TODO
        # base = "gpu_"
        # for i in range(2, 7):
        #     curr = base + str(i)
        #     assert checkpoints[curr] is None

    def test_path_types(self, pytorch_conf):
        base = "gpu_"
        for i in range(0, 8):
            copy = pytorch_conf.copy()
            curr = base + str(i)
            copy[curr] = 1234
            with pytest.raises(SchemaError):
                ModelSchema(copy)

    def test_missing_seed(self, pytorch_conf):
        pytorch_conf.pop(Keys.SEED.value)
        res = ModelSchema(pytorch_conf).val
        assert res[Keys.SEED.value] == 1231

    def test_missing_dimensions(self, pytorch_conf):
        pytorch_conf.pop(Keys.DIMENSIONS.value)
        with pytest.raises(SchemaError):
            ModelSchema(pytorch_conf)

    def test_missing_invocation(self, pytorch_conf):
        pytorch_conf.pop(Keys.INVOCATION.value)
        with pytest.raises(SchemaError):
            ModelSchema(pytorch_conf)

    def test_missing_execution(self, pytorch_conf):
        pytorch_conf.pop(Keys.EXECUTION.value)
        res = ModelSchema(pytorch_conf).val
        assert res[Keys.EXECUTION.value] == {
            Keys.HALF_PRECISION.value: True,
            Keys.JIT.value: True,
        }

    # TODO Do we want a schema error or a runtime or value error?
    # def test_no_paths_set(self, pytorch_conf):
    #     # pytorch_conf[Keys.CHECKPOINTS.value].pop(Keys.GPU_0.value)
    #     v = pytorch_conf[Keys.DEVICES.value].pop()
    #     print(v)
    #     # pytorch_conf[Keys.CHECKPOINTS.value].pop(Keys.GPU_1.value)
    #     with pytest.raises(SchemaError):
    #         ModelSchema(pytorch_conf)


class TestFieldsSchema:
    def test_normal_case(self, fields_conf):
        FieldsSchema(fields_conf)

    def test_bad_types(self, fields_conf):
        for i in range(len(fields_conf)):
            copy = fields_conf.copy()
            copy[i] = 12
            with pytest.raises(SchemaError):
                FieldsSchema(copy)


class TestSlidersSchema:
    def test_normal_case(self, sliders_conf):
        SlidersSchema(sliders_conf)

    def test_entry_not_dict(self, sliders_conf):
        sliders_conf["bad"] = "this is not a dictionary"
        with pytest.raises(SchemaError):
            SlidersSchema(sliders_conf)

    def test_bad_key(self, sliders_conf):
        sliders_conf[(1, 2, 3)] = sliders_conf.pop("a")
        with pytest.raises(SchemaWrongKeyError):
            SlidersSchema(sliders_conf)

    def test_missing_lower_bound(self, sliders_conf):
        sliders_conf["a"].pop(Keys.LOWER_BOUND.value)
        with pytest.raises(SchemaError):
            SlidersSchema(sliders_conf)

    def test_bad_lower_bound_type(self, sliders_conf):
        tests = ["str", {"a": 2}, [1, 2, 3]]
        for i in tests:
            copy = sliders_conf.copy()
            sliders_conf["a"][Keys.LOWER_BOUND.value] = i
            with pytest.raises(SchemaError):
                SlidersSchema(copy)

    def test_missing_upper(self, sliders_conf):
        sliders_conf["a"].pop(Keys.UPPER_BOUND.value)
        with pytest.raises(SchemaError):
            SlidersSchema(sliders_conf)

    def test_bad_upper_bound_type(self, sliders_conf):
        tests = ["str", {"a": 2}, [1, 2, 3]]
        for i in tests:
            copy = sliders_conf.copy()
            sliders_conf["a"][Keys.UPPER_BOUND.value] = i
            with pytest.raises(SchemaError):
                SlidersSchema(copy)

    def test_missing_initial_value(self, sliders_conf):
        sliders_conf["a"].pop(Keys.INITIAL_VALUE.value)
        res = SlidersSchema(sliders_conf).val
        assert res["a"][Keys.INITIAL_VALUE.value] == Keys.MEAN.value

    def test_bad_initial_value_type(self, sliders_conf):
        tests = ["str", {"a": 2}, [1, 2, 3]]
        for i in tests:
            copy = sliders_conf.copy()
            sliders_conf["a"][Keys.INITIAL_VALUE.value] = i
            with pytest.raises(SchemaError):
                SlidersSchema(copy)


class TestApperanceSchema:
    def test_normal_case(self, appearance_conf):
        AppearanceSchema(appearance_conf)

    def test_colormap_wrong_value(self, appearance_conf):
        appearance_conf[Keys.COLORMAP.value] = 2
        with pytest.raises(SchemaError):
            AppearanceSchema(appearance_conf)

    def test_theme_wrong_value(self, appearance_conf):
        appearance_conf[Keys.THEME.value] = 2
        with pytest.raises(SchemaError):
            AppearanceSchema(appearance_conf)

    def test_invalid_theme_value(self, appearance_conf):
        appearance_conf[Keys.THEME.value] = "not valid"
        with pytest.raises(SchemaError):
            AppearanceSchema(appearance_conf)

    def test_missing_colormap(self, appearance_conf):
        appearance_conf.pop(Keys.COLORMAP.value)
        res = AppearanceSchema(appearance_conf).val
        assert res[Keys.COLORMAP.value] == AppearanceSchema.DEFAULT_COLORMAP

    def test_missing_theme(self, appearance_conf):
        appearance_conf.pop(Keys.THEME.value)
        res = AppearanceSchema(appearance_conf).val
        assert res[Keys.THEME.value] == AppearanceSchema.DEFAULT_THEME

    def test_missing_all_keys(self, appearance_conf):
        appearance_conf.pop(Keys.COLORMAP.value)
        appearance_conf.pop(Keys.THEME.value)
        res = AppearanceSchema(appearance_conf).val
        assert len(res) == 2
        assert res[Keys.COLORMAP.value] == AppearanceSchema.DEFAULT_COLORMAP
        assert res[Keys.THEME.value] == AppearanceSchema.DEFAULT_THEME


class TestPluginsSchema:
    def test_normal_case(self, plugins_conf):
        # There is a special case with plugins because of not checking extra keys
        res = PluginsSchema(plugins_conf).val
        assert res["analytical"]["settings"]

    def test_bad_plugin_name(self, plugins_conf):
        ans = plugins_conf.get("analytical")
        plugins_conf[2] = ans
        with pytest.raises(SchemaWrongKeyError):
            PluginsSchema(plugins_conf)

    def test_missing_auto_load(self, plugins_conf):
        plugins_conf["analytical"].pop(Keys.AUTO_LOAD.value)
        res = PluginsSchema(plugins_conf).val
        assert not res["analytical"][Keys.AUTO_LOAD.value]


class TestGUISchema:
    def test_normal_case(self, napari_conf):
        GUISchema(napari_conf)

    def test_missing_fields(self, napari_conf):
        napari_conf.pop(Keys.FIELDS.value)
        with pytest.raises(SchemaMissingKeyError):
            GUISchema(napari_conf)

    def test_missing_sliders(self, napari_conf):
        napari_conf.pop(Keys.SLIDERS.value)
        with pytest.raises(SchemaMissingKeyError):
            GUISchema(napari_conf)

    def test_missing_appearance(self, napari_conf):
        napari_conf.pop(Keys.APPEARANCE.value)
        res = GUISchema(napari_conf).val
        assert res[Keys.APPEARANCE.value] == {
            Keys.COLORMAP.value: AppearanceSchema.DEFAULT_COLORMAP,
            Keys.THEME.value: AppearanceSchema.DEFAULT_THEME,
        }

    def test_missing_plugins(self, napari_conf):
        napari_conf.pop(Keys.PLUGINS.value)
        res = GUISchema(napari_conf).val
        assert res[Keys.PLUGINS.value] is None


class TestFullConf:
    def test_normal_case(self, conf):
        FullSchema(conf)

    def test_missing_model(self, conf):
        conf.pop(Keys.MODEL.value)
        with pytest.raises(SchemaError):
            FullSchema(conf)

    def test_missing_pytorch(self, conf):
        conf[Keys.MODEL.value].pop(Keys.PYTORCH.value)
        with pytest.raises(SchemaError):
            FullSchema(conf)

    def test_wrong_model_key(self, conf):
        res = conf.pop(Keys.MODEL.value)
        conf["wrong model type"] = res
        with pytest.raises(SchemaMissingKeyError):
            FullSchema(conf)

    def test_missing_gui(self, conf):
        conf.pop(Keys.GUI.value)
        with pytest.raises(SchemaMissingKeyError):
            FullSchema(conf)

    def test_missing_napari(self, conf):
        conf[Keys.GUI.value].pop(Keys.NAPARI.value)
        with pytest.raises(SchemaError):
            FullSchema(conf)

    def test_wrong_gui_key(self, conf):
        res = conf.pop(Keys.GUI.value)
        conf["wrong gui type"] = res
        with pytest.raises(SchemaMissingKeyError):
            FullSchema(conf)


class TestConfig:
    def test_full_set(self, config_mock, validated_conf):
        assert config_mock.full == validated_conf

    def test_get_top_level_key_name(self, config_mock):
        with pytest.raises(ValueError):
            config_mock.get_top_level_key_name()
        with pytest.raises(ValueError):
            config_mock.get_top_level_key_name(model=True, gui=True)

    def test_get_type_string_valid(self, config_mock):
        assert Keys.PYTORCH.value == config_mock.get_top_level_key_name(model=True)
        assert Keys.NAPARI.value == config_mock.get_top_level_key_name(gui=True)

    def test_model_set(self, config_mock, validated_conf):
        assert config_mock.model == validated_conf[Keys.MODEL.value][Keys.PYTORCH.value]

    def test_checkpoints_set(self, config_mock, validated_conf):
        assert config_mock.checkpoints == validated_conf[Keys.MODEL.value][Keys.PYTORCH.value][Keys.CHECKPOINTS.value]

    # def test_gpu_0_set(self, config_mock, validated_conf):
    #     assert (
    #         config_mock.gpu_0
    #         == validated_conf[Keys.MODEL.value][Keys.PYTORCH.value][
    #             Keys.CHECKPOINTS.value
    #         ][Keys.GPU_0.value]
    #     )

    # def test_gpu_1_set(self, config_mock, validated_conf):
    #     assert (
    #         config_mock.gpu_1
    #         == validated_conf[Keys.MODEL.value][Keys.PYTORCH.value][
    #             Keys.CHECKPOINTS.value
    #         ][Keys.GPU_1.value]
    #     )

    # def test_gpu_2_set(self, config_mock, validated_conf):
    #     assert (
    #         config_mock.gpu_2
    #         == validated_conf[Keys.MODEL.value][Keys.PYTORCH.value][
    #             Keys.CHECKPOINTS.value
    #         ][Keys.GPU_2.value]
    #     )

    # def test_gpu_3_set(self, config_mock, validated_conf):
    #     assert (
    #         config_mock.gpu_3
    #         == validated_conf[Keys.MODEL.value][Keys.PYTORCH.value][
    #             Keys.CHECKPOINTS.value
    #         ][Keys.GPU_3.value]
    #     )

    # def test_gpu_4_set(self, config_mock, validated_conf):
    #     assert (
    #         config_mock.gpu_4
    #         == validated_conf[Keys.MODEL.value][Keys.PYTORCH.value][
    #             Keys.CHECKPOINTS.value
    #         ][Keys.GPU_4.value]
    #     )

    # def test_gpu_5_set(self, config_mock, validated_conf):
    #     assert (
    #         config_mock.gpu_5
    #         == validated_conf[Keys.MODEL.value][Keys.PYTORCH.value][
    #             Keys.CHECKPOINTS.value
    #         ][Keys.GPU_5.value]
    #     )

    # def test_gpu_6_set(self, config_mock, validated_conf):
    #     assert (
    #         config_mock.gpu_6
    #         == validated_conf[Keys.MODEL.value][Keys.PYTORCH.value][
    #             Keys.CHECKPOINTS.value
    #         ][Keys.GPU_6.value]
    #     )

    # def test_gpu_7_set(self, config_mock, validated_conf):
    #     assert (
    #         config_mock.gpu_7
    #         == validated_conf[Keys.MODEL.value][Keys.PYTORCH.value][
    #             Keys.CHECKPOINTS.value
    #         ][Keys.GPU_7.value]
    #     )

    def test_seed_set(self, config_mock, validated_conf):
        assert config_mock.seed == validated_conf[Keys.MODEL.value][Keys.PYTORCH.value][Keys.SEED.value]

    def test_dimensions_Set(self, config_mock, validated_conf):
        assert config_mock.dimensions == validated_conf[Keys.MODEL.value][Keys.PYTORCH.value][Keys.DIMENSIONS.value]

    def test_n_inputs_set(self, config_mock, validated_conf):
        assert (
            config_mock.n_inputs
            == validated_conf[Keys.MODEL.value][Keys.PYTORCH.value][Keys.DIMENSIONS.value][Keys.N_INPUTS.value]
        )

    def test_batch_size_set(self, config_mock, validated_conf):
        assert (
            config_mock.batch_size
            == validated_conf[Keys.MODEL.value][Keys.PYTORCH.value][Keys.DIMENSIONS.value][Keys.BATCH_SIZE.value]
        )

    def test_y_pixels_set(self, config_mock, validated_conf):
        assert (
            config_mock.y_pixels
            == validated_conf[Keys.MODEL.value][Keys.PYTORCH.value][Keys.DIMENSIONS.value][Keys.Y_PIXELS.value]
        )

    def test_x_pixels_set(self, config_mock, validated_conf):
        assert (
            config_mock.x_pixels
            == validated_conf[Keys.MODEL.value][Keys.PYTORCH.value][Keys.DIMENSIONS.value][Keys.X_PIXELS.value]
        )

    def test_invocation_set(self, config_mock, validated_conf):
        assert config_mock.invocation == validated_conf[Keys.MODEL.value][Keys.PYTORCH.value][Keys.INVOCATION.value]

    def test_invocation_target_set(self, config_mock, validated_conf):
        assert (
            config_mock.invocation_target
            == validated_conf[Keys.MODEL.value][Keys.PYTORCH.value][Keys.INVOCATION.value][Keys.TARGET.value]
        )

    def test_invocation_params_set(self, config_mock, validated_conf):
        assert (
            config_mock.invocation_params
            == validated_conf[Keys.MODEL.value][Keys.PYTORCH.value][Keys.INVOCATION.value][Keys.PARAMS.value]
        )

    def test_execution_set(self, config_mock, validated_conf):
        assert config_mock.execution == validated_conf[Keys.MODEL.value][Keys.PYTORCH.value][Keys.EXECUTION.value]

    def test_half_precision_set(self, config_mock, validated_conf):
        assert {
            config_mock.half_precision
            == validated_conf[Keys.MODEL.value][Keys.PYTORCH.value][Keys.EXECUTION.value][Keys.HALF_PRECISION.value]
        }

    def test_jit_set(self, config_mock, validated_conf):
        assert {
            config_mock.jit
            == validated_conf[Keys.MODEL.value][Keys.PYTORCH.value][Keys.EXECUTION.value][Keys.JIT.value]
        }

    def test_gui_set(self, config_mock, validated_conf):
        assert config_mock.gui == validated_conf[Keys.GUI.value][Keys.NAPARI.value]

    def test_fields_set(self, config_mock, validated_conf):
        assert config_mock.fields == validated_conf[Keys.GUI.value][Keys.NAPARI.value][Keys.FIELDS.value]

    def test_get_field_map(self, config_mock):
        expected = {"materials": 0}
        assert expected == config_mock.get_field_map()
        assert expected == config_mock.field_map

    def test_original_sliders_set(self, config_mock, validated_conf):
        assert config_mock.original_sliders == validated_conf[Keys.GUI.value][Keys.NAPARI.value][Keys.SLIDERS.value]

    def test_evaluate_sliders(self, config_mock, sliders_conf):
        sliders_conf["a"][Keys.INITIAL_VALUE.value] = 0.55
        sliders_conf["b"][Keys.INITIAL_VALUE.value] = 0.55
        sliders_conf["n"][Keys.INITIAL_VALUE.value] = 3
        assert config_mock.evaluate_sliders() == sliders_conf
        assert config_mock.sliders == config_mock.evaluate_sliders()

    def test_get_slider_initial_values(self, config_mock):
        expected = [0.55, 0.55, 3, 0]
        assert config_mock.get_slider_initial_values() == expected

    def test_appearance_set(self, config_mock, validated_conf):
        assert config_mock.appearance == validated_conf[Keys.GUI.value][Keys.NAPARI.value][Keys.APPEARANCE.value]

    def test_plugins_set(self, config_mock, validated_conf):
        assert config_mock.plugins == validated_conf[Keys.GUI.value][Keys.NAPARI.value][Keys.PLUGINS.value]

    def test_multiple_checkpoints_exist(self, config_mock):
        assert config_mock.multiple_checkpoints_exist()

    def test_multiple_checkpoints_exist_fail(self, config_mock):
        config_mock.loaded_checkpoints.pop()
        assert not config_mock.multiple_checkpoints_exist()

    # string or int, this should not matter...
    # def test_get_device_type(self, config_mock):
    #     expected = "0"
    #     assert expected == config_mock.get_device_type_from_string("gpu_0")
