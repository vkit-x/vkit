# Copyright 2022 vkit-x Administrator. All Rights Reserved.
#
# This project (vkit-x/vkit) is dual-licensed under commercial and SSPL licenses.
#
# The commercial license gives you the full rights to create and distribute software
# on your own terms without any SSPL license obligations. For more information,
# please see the "LICENSE_COMMERCIAL.txt" file.
#
# This project is also available under Server Side Public License (SSPL).
# The SSPL licensing is ideal for use cases such as open source projects with
# SSPL distribution, student/academic purposes, hobby projects, internal research
# projects without external distribution, or other projects where all SSPL
# obligations can be met. For more information, please see the "LICENSE_SSPL.txt" file.
from typing import (
    cast,
    Generic,
    TypeVar,
    Type,
    Dict,
    Mapping,
    Any,
    Sequence,
    Optional,
    Union,
    List,
)

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.utility import (
    is_path_type,
    read_json_file,
    dyn_structure,
    convert_camel_case_name_to_snake_case_name,
    get_generic_classes,
    PathType,
)

_T_VALUE = TypeVar('_T_VALUE')
_T_CONFIG = TypeVar('_T_CONFIG')
_T_INPUT = TypeVar('_T_INPUT')
_T_OUTPUT = TypeVar('_T_OUTPUT')


@attrs.define
class PipelineState:
    key_to_value: Dict[str, Any] = attrs.field(factory=dict)

    def get_value(self, key: str, value_cls: Type[_T_VALUE]) -> _T_VALUE:
        if key not in self.key_to_value:
            raise KeyError(f'key={key} not found.')
        value = self.key_to_value[key]
        if not isinstance(value, value_cls):
            raise TypeError(f'key={key}, value type={type(value)} is not instance of {value_cls}')
        return value

    def set_value(self, key: str, value: Any, override: bool = False):
        if key in self.key_to_value and not override:
            raise KeyError(f'key={key} exists but override is not set.')
        self.key_to_value[key] = value


class PipelineStep(Generic[_T_CONFIG, _T_INPUT, _T_OUTPUT]):

    @classmethod
    def get_config_cls(cls) -> Type[_T_CONFIG]:
        return get_generic_classes(cls)[0]  # type: ignore

    @classmethod
    def get_input_cls(cls) -> Type[_T_INPUT]:
        return get_generic_classes(cls)[1]  # type: ignore

    @classmethod
    def get_output_cls(cls) -> Type[_T_OUTPUT]:
        return get_generic_classes(cls)[2]  # type: ignore

    _cached_name: str = ''

    @classmethod
    def get_name(cls):
        if not cls._cached_name:
            cls._cached_name = convert_camel_case_name_to_snake_case_name(cls.__name__)
        return cls._cached_name

    def __init__(self, config: _T_CONFIG):
        self.config = config

    def run(self, input: _T_INPUT, rng: RandomGenerator) -> _T_OUTPUT:
        raise NotImplementedError()


class PipelineStepFactory(Generic[_T_CONFIG, _T_INPUT, _T_OUTPUT]):

    def __init__(self, pipeline_step_cls: Type[PipelineStep[_T_CONFIG, _T_INPUT, _T_OUTPUT]]):
        self.pipeline_step_cls = pipeline_step_cls

    @property
    def name(self):
        return self.pipeline_step_cls.get_name()

    def get_config_cls(self):
        return self.pipeline_step_cls.get_config_cls()

    def create(
        self,
        config: Optional[Union[Mapping[str, Any], PathType, _T_CONFIG]] = None,
    ):
        config = dyn_structure(
            config,
            self.get_config_cls(),
            support_path_type=True,
            support_none_type=True,
        )
        return self.pipeline_step_cls(config)


class PipelineStepCollectionFactory:

    def __init__(self):
        self.name_to_step_factory: Dict[str, PipelineStepFactory] = {}

    def register_step_factories(
        self,
        namespace: str,
        step_factories: Sequence[PipelineStepFactory],
    ):
        for step_factory in step_factories:
            name = f'{namespace}.{step_factory.name}'
            assert name not in self.name_to_step_factory
            self.name_to_step_factory[name] = step_factory

    def create(
        self,
        step_configs: Union[Sequence[Mapping[str, Any]], PathType],
    ):
        if is_path_type(step_configs):
            step_configs = read_json_file(step_configs)  # type: ignore
        step_configs = cast(Sequence[Mapping[str, Any]], step_configs)

        steps: List[PipelineStep] = []
        for step_config in step_configs:
            name = step_config['name']
            if name not in self.name_to_step_factory:
                raise KeyError(f'name={name} not found.')
            step_factory = self.name_to_step_factory[name]
            steps.append(step_factory.create(step_config.get('config')))
        return steps


class PipelinePostProcessor(Generic[_T_CONFIG, _T_INPUT, _T_OUTPUT]):

    def __init__(self, config: _T_CONFIG):
        self.config = config

    @classmethod
    def get_input_cls(cls) -> Type[_T_INPUT]:
        return get_generic_classes(cls)[1]  # type: ignore

    def generate_output(self, input: _T_INPUT, rng: RandomGenerator) -> _T_OUTPUT:
        raise NotImplementedError()


class PipelinePostProcessorFactory(Generic[_T_CONFIG, _T_INPUT, _T_OUTPUT]):

    def __init__(
        self,
        pipeline_post_processor_cls: Type[PipelinePostProcessor[_T_CONFIG, _T_INPUT, _T_OUTPUT]],
    ):
        self.pipeline_post_processor_cls = pipeline_post_processor_cls

    def get_config_cls(self) -> Type[_T_CONFIG]:
        return get_generic_classes(self.pipeline_post_processor_cls)[0]  # type: ignore

    def create(
        self,
        config: Optional[Union[Mapping[str, Any], PathType, _T_CONFIG]] = None,
    ):
        config = dyn_structure(
            config,
            self.get_config_cls(),
            support_path_type=True,
            support_none_type=True,
        )
        return self.pipeline_post_processor_cls(config)


@attrs.define
class PipelineRunRngStateOutput:
    rng_state: Mapping[str, Any]


class Pipeline(Generic[_T_OUTPUT]):

    def __init__(
        self,
        steps: Sequence[PipelineStep],
        post_processor: PipelinePostProcessor[Any, Any, _T_OUTPUT],
    ):
        self.steps = steps
        self.post_processor = post_processor

    @classmethod
    def build_input(cls, state: PipelineState, input_cls: Any):
        assert attrs.has(input_cls)

        input_kwargs = {}
        for key, key_field in attrs.fields_dict(input_cls).items():
            assert key_field.type
            assert attrs.has(key_field.type)
            value = state.get_value(
                convert_camel_case_name_to_snake_case_name(key_field.type.__name__),
                key_field.type,
            )
            input_kwargs[key] = value

        return input_cls(**input_kwargs)

    def run(
        self,
        rng: RandomGenerator,
        state: Optional[PipelineState] = None,
    ) -> _T_OUTPUT:
        if state is None:
            state = PipelineState()

        # Save the rng state.
        state.set_value(
            convert_camel_case_name_to_snake_case_name(PipelineRunRngStateOutput.__name__),
            PipelineRunRngStateOutput(rng.bit_generator.state),
        )

        # Run steps.
        for step in self.steps:
            # Build input.
            step_input = self.build_input(state, step.get_input_cls())

            # Generate output.
            step_output = step.run(step_input, rng)

            # Update state.
            step_output_cls = step.get_output_cls()
            assert isinstance(step_output, step_output_cls)
            assert attrs.has(step_output_cls)
            state.set_value(
                convert_camel_case_name_to_snake_case_name(step_output_cls.__name__),
                step_output,
            )

        # Post processing.
        return self.post_processor.generate_output(
            self.build_input(state, self.post_processor.get_input_cls()),
            rng,
        )
