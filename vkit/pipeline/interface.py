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

    def get_pipeline_step_output(
        self,
        pipeline_step: 'Type[PipelineStep[Any, _T_OUTPUT]]',
    ) -> _T_OUTPUT:
        return self.get_value(
            pipeline_step.get_name(),
            pipeline_step.get_output_cls(),
        )


@attrs.define
class NoneTypePipelineStepConfig:
    pass


class PipelineStep(Generic[_T_CONFIG, _T_OUTPUT]):

    @classmethod
    def get_config_cls(cls) -> Type[_T_CONFIG]:
        return get_generic_classes(cls)[0]  # type: ignore

    @classmethod
    def get_output_cls(cls) -> Type[_T_OUTPUT]:
        return get_generic_classes(cls)[1]  # type: ignore

    _cached_name: str = ''

    @classmethod
    def get_name(cls):
        if not cls._cached_name:
            cls._cached_name = convert_camel_case_name_to_snake_case_name(cls.__name__)
        return cls._cached_name

    def __init__(self, config: _T_CONFIG):
        self.config = config

    def run(self, state: PipelineState, rng: RandomGenerator) -> _T_OUTPUT:
        raise NotImplementedError()


class PipelineStepFactory(Generic[_T_CONFIG, _T_OUTPUT]):

    def __init__(self, pipeline_step_cls: Type[PipelineStep[_T_CONFIG, _T_OUTPUT]]):
        self.pipeline_step_cls = pipeline_step_cls

    @property
    def name(self):
        return self.pipeline_step_cls.get_name()

    def get_config_cls(self) -> Type[_T_CONFIG]:
        return get_generic_classes(self.pipeline_step_cls)[0]  # type: ignore

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


@attrs.define
class NoneTypePipelinePostProcessorConfig:
    pass


class PipelinePostProcessor(Generic[_T_CONFIG, _T_OUTPUT]):

    def __init__(self, config: _T_CONFIG):
        self.config = config

    def generate_output(
        self,
        state: PipelineState,
        rng: RandomGenerator,
    ) -> _T_OUTPUT:
        raise NotImplementedError()


class BypassPipelinePostProcessor(
    PipelinePostProcessor[
        NoneTypePipelinePostProcessorConfig,
        PipelineState,
    ]
):  # yapf: disable

    def generate_output(self, state: PipelineState, rng: RandomGenerator) -> PipelineState:
        return state


class PipelinePostProcessorFactory(Generic[_T_CONFIG, _T_OUTPUT]):

    def __init__(
        self,
        pipeline_post_processor_cls: Type[PipelinePostProcessor[_T_CONFIG, _T_OUTPUT]],
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


bypass_post_processor_factory = PipelinePostProcessorFactory(BypassPipelinePostProcessor)


class Pipeline(Generic[_T_OUTPUT]):

    def __init__(
        self,
        steps: Sequence[PipelineStep],
        post_processor: PipelinePostProcessor[Any, _T_OUTPUT],
    ):
        self.steps = steps
        self.post_processor = post_processor

    def run(
        self,
        rng: RandomGenerator,
        state: Optional[PipelineState] = None,
    ) -> _T_OUTPUT:
        if state is None:
            state = PipelineState()
        for step in self.steps:
            output = step.run(state, rng)
            state.set_value(step.get_name(), output)
        return self.post_processor.generate_output(state, rng)
