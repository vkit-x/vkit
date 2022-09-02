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
    Type,
    TypeVar,
    Mapping,
    Sequence,
    Any,
    Tuple,
    List,
    Optional,
    Union,
    Callable,
)
import itertools

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.utility import (
    normalize_to_keys_and_probs,
    rng_choice,
    is_path_type,
    read_json_file,
    dyn_structure,
    get_generic_classes,
    PathType,
)

_T_INIT_CONFIG = TypeVar('_T_INIT_CONFIG')
_T_INIT_RESOURCE = TypeVar('_T_INIT_RESOURCE')
_T_RUN_CONFIG = TypeVar('_T_RUN_CONFIG')
_T_RUN_OUTPUT = TypeVar('_T_RUN_OUTPUT')


@attrs.define
class NoneTypeEngineInitConfig:
    pass


@attrs.define
class NoneTypeEngineInitResource:
    pass


class Engine(
    Generic[
        _T_INIT_CONFIG,
        _T_INIT_RESOURCE,
        _T_RUN_CONFIG,
        _T_RUN_OUTPUT,
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        raise NotImplementedError()

    def __init__(
        self,
        init_config: _T_INIT_CONFIG,
        init_resource: Optional[_T_INIT_RESOURCE] = None,
    ):
        self.init_config = init_config
        self.init_resource = init_resource

    def run(self, run_config: _T_RUN_CONFIG, rng: RandomGenerator) -> _T_RUN_OUTPUT:
        raise NotImplementedError()


class EngineExecutor(
    Generic[
        _T_INIT_CONFIG,
        _T_INIT_RESOURCE,
        _T_RUN_CONFIG,
        _T_RUN_OUTPUT,
    ]
):  # yapf: disable

    def __init__(
        self,
        engine: Engine[
            _T_INIT_CONFIG,
            _T_INIT_RESOURCE,
            _T_RUN_CONFIG,
            _T_RUN_OUTPUT,
        ],
    ):  # yapf: disable
        self.engine = engine

    def get_run_config_cls(self) -> Type[_T_RUN_CONFIG]:
        return get_generic_classes(type(self.engine))[2]  # type: ignore

    def run(
        self,
        run_config: Union[
            Mapping[str, Any],
            _T_RUN_CONFIG,
        ],
        rng: RandomGenerator,
    ) -> _T_RUN_OUTPUT:  # yapf: disable
        run_config = dyn_structure(run_config, self.get_run_config_cls())
        return self.engine.run(run_config, rng)


class EngineExecutorFactory(
    Generic[
        _T_INIT_CONFIG,
        _T_INIT_RESOURCE,
        _T_RUN_CONFIG,
        _T_RUN_OUTPUT,
    ]
):  # yapf: disable

    def __init__(
        self,
        engine_cls: Type[
            Engine[
                _T_INIT_CONFIG,
                _T_INIT_RESOURCE,
                _T_RUN_CONFIG,
                _T_RUN_OUTPUT,
            ]
        ],
    ):  # yapf: disable
        self.engine_cls = engine_cls

    def get_type_name(self):
        return self.engine_cls.get_type_name()

    def get_init_config_cls(self) -> Type[_T_INIT_CONFIG]:
        return get_generic_classes(self.engine_cls)[0]  # type: ignore

    def get_init_resource_cls(self) -> Type[_T_INIT_RESOURCE]:
        return get_generic_classes(self.engine_cls)[1]  # type: ignore

    def create(
        self,
        init_config: Optional[
            Union[
                Mapping[str, Any],
                PathType,
                _T_INIT_CONFIG,
            ]
        ] = None,
        init_resource: Optional[
            Union[
                Mapping[str, Any],
                _T_INIT_RESOURCE,
            ]
        ] = None,
    ):  # yapf: disable
        init_config = dyn_structure(
            init_config,
            self.get_init_config_cls(),
            support_path_type=True,
            support_none_type=True,
        )

        init_resource_cls = self.get_init_resource_cls()
        if init_resource_cls is NoneTypeEngineInitResource:
            assert init_resource is None
        else:
            assert init_resource
        if init_resource is not None:
            init_resource = dyn_structure(init_resource, init_resource_cls)

        return EngineExecutor(self.engine_cls(init_config, init_resource))


class EngineExecutorAggregatorSelector(
    Generic[
        _T_RUN_CONFIG,
        _T_RUN_OUTPUT,
    ]
):  # yapf: disable

    def __init__(
        self,
        pairs: Sequence[
            Tuple[
                EngineExecutor[
                    Any,
                    Any,
                    _T_RUN_CONFIG,
                    _T_RUN_OUTPUT,
                ],
                float,
            ]
        ],
    ):  # yapf: disable
        self.engine_executors, self.probs = normalize_to_keys_and_probs(pairs)

    def get_run_config_cls(self):
        return self.engine_executors[0].get_run_config_cls()

    def select_engine_executor(self, rng: RandomGenerator):
        return rng_choice(rng, self.engine_executors, probs=self.probs)


def engine_executor_aggregator_default_func_collate(
    selector: EngineExecutorAggregatorSelector[
        _T_RUN_CONFIG,
        _T_RUN_OUTPUT,
    ],
    run_config: _T_RUN_CONFIG,
    rng: RandomGenerator,
) -> _T_RUN_OUTPUT:  # yapf: disable
    engine_executor = selector.select_engine_executor(rng)
    return engine_executor.run(run_config, rng)


class EngineExecutorAggregator(
    Generic[
        _T_RUN_CONFIG,
        _T_RUN_OUTPUT,
    ]
):  # yapf: disable

    def get_run_config_cls(self):
        return self.selector.get_run_config_cls()

    def __init__(
        self,
        selector: EngineExecutorAggregatorSelector[
            _T_RUN_CONFIG,
            _T_RUN_OUTPUT,
        ],
        func_collate: Callable[
            [
                EngineExecutorAggregatorSelector[
                    _T_RUN_CONFIG,
                    _T_RUN_OUTPUT,
                ],
                _T_RUN_CONFIG,
                RandomGenerator,
            ],
            _T_RUN_OUTPUT,
        ] = engine_executor_aggregator_default_func_collate,
    ):  # yapf: disable
        self.selector = selector
        self.func_collate = func_collate

    def run(
        self,
        run_config: Union[
            Mapping[str, Any],
            _T_RUN_CONFIG,
        ],
        rng: RandomGenerator,
    ) -> _T_RUN_OUTPUT:  # yapf: disable
        run_config = dyn_structure(run_config, self.get_run_config_cls())
        return self.func_collate(self.selector, run_config, rng)


class EngineExecutorAggregatorFactoryConfigKey:
    TYPE = 'type'
    WEIGHT = 'weight'
    CONFIG = 'config'


class EngineExecutorAggregatorFactory(
    Generic[
        _T_RUN_CONFIG,
        _T_RUN_OUTPUT,
    ]
):  # yapf: disable

    def __init__(
        self,
        engine_executor_factories: Sequence[
            EngineExecutorFactory[
                Any,
                Any,
                _T_RUN_CONFIG,
                _T_RUN_OUTPUT,
            ]
        ],
        func_collate: Callable[
            [
                EngineExecutorAggregatorSelector[
                    _T_RUN_CONFIG,
                    _T_RUN_OUTPUT,
                ],
                _T_RUN_CONFIG,
                RandomGenerator,
            ],
            _T_RUN_OUTPUT,
        ] = engine_executor_aggregator_default_func_collate,
    ):  # yapf: disable
        self.type_name_to_engine_executor_factory = {
            engine_executor_factory.get_type_name(): engine_executor_factory
            for engine_executor_factory in engine_executor_factories
        }
        self.func_collate = func_collate

    def create(
        self,
        factory_init_configs: Union[
            Sequence[Mapping[str, Any]],
            PathType,
        ],
        init_resources: Optional[
            Sequence[
                Union[
                    Mapping[str, Any],
                    # TODO: find a better way to constrain resource type.
                    Any,
                ]
            ]
        ] = None,
    ):  # yapf: disable
        if is_path_type(factory_init_configs):
            factory_init_configs = read_json_file(factory_init_configs)  # type: ignore
        factory_init_configs = cast(Sequence[Mapping[str, Any]], factory_init_configs)

        pairs: List[
            Tuple[
                EngineExecutor[Any, Any, _T_RUN_CONFIG, _T_RUN_OUTPUT],
                float,
            ],
        ] = []  # yapf: disable

        for factory_init_config, init_resource in zip(
            factory_init_configs, init_resources or itertools.repeat(None)
        ):
            # Reflects engine_executor_factory.
            type_name = factory_init_config[EngineExecutorAggregatorFactoryConfigKey.TYPE]
            if type_name not in self.type_name_to_engine_executor_factory:
                raise KeyError(f'type_name={type_name} not found')
            engine_executor_factory = self.type_name_to_engine_executor_factory[type_name]

            # Build init_resource.
            init_resource_cls = engine_executor_factory.get_init_resource_cls()
            if init_resource_cls is NoneTypeEngineInitResource:
                assert init_resource is None
            else:
                assert init_resource
                init_resource = dyn_structure(init_resource, init_resource_cls)

            # Build engine_executor.
            engine_executor = engine_executor_factory.create(
                factory_init_config.get(EngineExecutorAggregatorFactoryConfigKey.CONFIG, {}),
                init_resource,
            )

            # Get the weight.
            if len(factory_init_configs) == 1:
                weight = 1
            else:
                weight = factory_init_config[EngineExecutorAggregatorFactoryConfigKey.WEIGHT]

            pairs.append((engine_executor, weight))

        return EngineExecutorAggregator(
            EngineExecutorAggregatorSelector(pairs),
            func_collate=self.func_collate,
        )

    def create_with_repeated_init_resource(
        self,
        factory_init_configs: Union[
            Sequence[Mapping[str, Any]],
            PathType,
        ],
        init_resource: Union[
            Mapping[str, Any],
            Any,
        ]
    ):  # yapf: disable
        if is_path_type(factory_init_configs):
            factory_init_configs = read_json_file(factory_init_configs)  # type: ignore
        factory_init_configs = cast(Sequence[Mapping[str, Any]], factory_init_configs)

        return self.create(
            factory_init_configs,
            [init_resource] * len(factory_init_configs),
        )


# TODO: move to doc.
# A minimum template.
'''
from typing import Optional

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.engine.interface import (
    Engine,
    EngineExecutorFactory,
    NoneTypeEngineInitResource,
)


@attrs.define
class FooEngineInitConfig:
    pass


@attrs.define
class FooEngineRunConfig:
    pass


class FooEngine(
    Engine[
        FooEngineInitConfig,
        NoneTypeEngineInitResource,
        FooEngineRunConfig,
        int,
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        return 'foo'

    def __init__(
        self,
        init_config: FooEngineInitConfig,
        init_resource: Optional[NoneTypeEngineInitResource] = None,
    ):
        super().__init__(init_config, init_resource)

    def run(self, run_config: FooEngineRunConfig, rng: RandomGenerator) -> int:
        return 42


foo_engine_executor_factory = EngineExecutorFactory(FooEngine)
'''
