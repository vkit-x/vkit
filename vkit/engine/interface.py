from typing import (
    cast,
    Generic,
    Type,
    TypeVar,
    Mapping,
    Sequence,
    Any,
    Tuple,
    List,
    Optional,
    Union,
)
from enum import Enum, unique

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.utility import (
    rng_choice,
    is_path_type,
    read_json_file,
    dyn_structure,
    get_generic_classes,
    PathType,
)

_T_CONFIG = TypeVar('_T_CONFIG')
_T_RESOURCE = TypeVar('_T_RESOURCE')
_T_RUN_CONFIG = TypeVar('_T_RUN_CONFIG')
_T_OUTPUT = TypeVar('_T_OUTPUT')


@attrs.define
class NoneTypeEngineConfig:
    pass


@attrs.define
class NoneTypeEngineResource:
    pass


class Engine(Generic[_T_CONFIG, _T_RESOURCE, _T_RUN_CONFIG, _T_OUTPUT]):

    @classmethod
    def get_type_name(cls) -> str:
        raise NotImplementedError()

    def __init__(
        self,
        config: _T_CONFIG,
        resource: Optional[_T_RESOURCE] = None,
    ):
        self.config = config
        self.resource = resource

    def run(self, config: _T_RUN_CONFIG, rng: RandomGenerator) -> _T_OUTPUT:
        raise NotImplementedError()


class EngineRunner(Generic[_T_CONFIG, _T_RESOURCE, _T_RUN_CONFIG, _T_OUTPUT]):

    def __init__(self, engine: Engine[_T_CONFIG, _T_RESOURCE, _T_RUN_CONFIG, _T_OUTPUT]) -> None:
        self.engine = engine

    def get_run_config_cls(self) -> Type[_T_RUN_CONFIG]:
        return get_generic_classes(type(self.engine))[2]  # type: ignore

    def run(
        self,
        config: Union[Mapping[str, Any], _T_RUN_CONFIG],
        rng: RandomGenerator,
    ) -> _T_OUTPUT:
        config = dyn_structure(config, self.get_run_config_cls())
        return self.engine.run(config, rng)


class EngineFactory(Generic[_T_CONFIG, _T_RESOURCE, _T_RUN_CONFIG, _T_OUTPUT]):

    def __init__(
        self, engine_cls: Type[Engine[_T_CONFIG, _T_RESOURCE, _T_RUN_CONFIG, _T_OUTPUT]]
    ) -> None:
        self.engine_cls = engine_cls

    def get_type_name(self):
        return self.engine_cls.get_type_name()

    def get_config_cls(self) -> Type[_T_CONFIG]:
        return get_generic_classes(self.engine_cls)[0]  # type: ignore

    def get_resource_cls(self) -> Type[_T_RESOURCE]:
        return get_generic_classes(self.engine_cls)[1]  # type: ignore

    def create(
        self,
        config: Optional[Union[Mapping[str, Any], PathType, _T_CONFIG]] = None,
        resource: Optional[Union[Mapping[str, Any], _T_RESOURCE]] = None,
    ):
        config = dyn_structure(
            config,
            self.get_config_cls(),
            support_path_type=True,
            support_none_type=True,
        )

        resource_cls = self.get_resource_cls()
        if resource_cls is NoneTypeEngineResource:
            assert resource is None
        else:
            assert resource
        if resource is not None:
            resource = dyn_structure(resource, resource_cls)

        return EngineRunner(self.engine_cls(config, resource))


class EngineRunnerAggregator(Generic[_T_RESOURCE, _T_RUN_CONFIG, _T_OUTPUT]):

    def __init__(
        self,
        pairs: Sequence[Tuple[EngineRunner[Any, _T_RESOURCE, _T_RUN_CONFIG, _T_OUTPUT], float]],
    ):
        self.engine_runners: List[EngineRunner[Any, _T_RESOURCE, _T_RUN_CONFIG, _T_OUTPUT]] = []
        weights = []
        for engine_runner, weight in pairs:
            self.engine_runners.append(engine_runner)
            weights.append(weight)
        total = sum(weights)
        self.probs = [weight / total for weight in weights]

    def run(
        self,
        config: Union[Mapping[str, Any], _T_RUN_CONFIG],
        rng: RandomGenerator,
    ) -> _T_OUTPUT:
        engine_runner = rng_choice(rng, self.engine_runners, probs=self.probs)
        return engine_runner.run(config, rng)


@unique
class EngineRunnerAggregatorFactoryConfigKey(Enum):
    TYPE = 'type'
    WEIGHT = 'weight'
    CONFIG = 'config'


class EngineRunnerAggregatorFactory(Generic[_T_RESOURCE, _T_RUN_CONFIG, _T_OUTPUT]):

    def __init__(
        self, engine_factories: Sequence[EngineFactory[Any, _T_RESOURCE, _T_RUN_CONFIG, _T_OUTPUT]]
    ):
        self.type_name_to_engine_factory = {
            engine_factory.get_type_name(): engine_factory for engine_factory in engine_factories
        }

    def get_resource_cls(self) -> Type[_T_RESOURCE]:
        engine_cls = next(iter(self.type_name_to_engine_factory.values()))
        return engine_cls.get_resource_cls()

    def create(
        self,
        configs: Union[Sequence[Mapping[str, Any]], PathType],
        resource: Optional[Union[Mapping[str, Any], _T_RESOURCE]] = None,
    ):
        resource_cls = self.get_resource_cls()
        if resource_cls is NoneTypeEngineResource:
            assert resource is None
        else:
            assert resource
        if resource is not None:
            resource = dyn_structure(resource, resource_cls)

        if is_path_type(configs):
            configs = read_json_file(configs)  # type: ignore
        configs = cast(Sequence[Mapping[str, Any]], configs)

        pairs: List[Tuple[EngineRunner[Any, _T_RESOURCE, _T_RUN_CONFIG, _T_OUTPUT], float]] = []
        for config in configs:
            type_name = config[EngineRunnerAggregatorFactoryConfigKey.TYPE.value]
            if type_name not in self.type_name_to_engine_factory:
                raise KeyError(f'type_name={type_name} not found')

            engine_factory = self.type_name_to_engine_factory[type_name]
            engine_runner = engine_factory.create(
                config.get(EngineRunnerAggregatorFactoryConfigKey.CONFIG.value, {}),
                resource,
            )
            if len(configs) == 1:
                weight = 1
            else:
                weight = config[EngineRunnerAggregatorFactoryConfigKey.WEIGHT.value]

            pairs.append((engine_runner, weight))
        return EngineRunnerAggregator(pairs)
