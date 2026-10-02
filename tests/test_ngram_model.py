from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy

from models.ngram.model import Model

Token = tuple[int, int, int, int]


def _token(pitch: int) -> Token:
    return (pitch, 1, 1, 1)


def _model(
    models: list[dict[tuple[Token, ...], Counter[Token]]],
    neighbors: dict[Token, list[tuple[Token, float]]] | None = None,
) -> Model:
    model = Model(loader=[], neighbors=neighbors, n=len(models) + 1)
    model.models = models
    return model


def test_remi_predict_uses_rare_lowest_order_distribution(monkeypatch) -> None:
    sparse_target = 70
    fallback_target = 71
    models = [
        {(60, 62): Counter({sparse_target: 1})},
        {(62,): Counter({fallback_target: 1})},
    ]
    model = Model(loader=[], n=3)
    model.models = models
    captured = {}

    def choose(population, weights):
        captured.update(zip(population, weights, strict=True))
        return [population[0]]

    monkeypatch.setattr("models.ngram.model.random.choices", choose)

    prediction = model.remi_predict((60, 62))

    assert prediction == [fallback_target]
    assert captured == {fallback_target: 1}


def test_remi_predict_unknown_context_does_not_mutate_models() -> None:
    models = [defaultdict(Counter), defaultdict(Counter)]
    model = Model(loader=[], n=3)
    model.models = models

    assert model.remi_predict((60, 62)) is None
    assert all(not counts for counts in models)


def test_note_table_predict_queries_complete_neighbor_contexts(monkeypatch) -> None:
    first = _token(60)
    second = _token(62)
    second_neighbor = _token(63)
    exact_target = _token(70)
    neighbor_target = _token(71)
    fallback_target = _token(72)
    models = [
        defaultdict(
            Counter,
            {
                (first, second): Counter({exact_target: 1}),
                (first, second_neighbor): Counter({neighbor_target: 9}),
            },
        ),
        {(second,): Counter({fallback_target: 100})},
    ]
    model = _model(models, {second: [(second_neighbor, 0.1)]})
    before = deepcopy(model.models)
    captured = {}

    def choose(population, weights):
        captured.update(zip(population, weights, strict=True))
        return [population[0]]

    monkeypatch.setattr("models.ngram.model.random.choices", choose)

    model.note_table_predict((first, second))

    assert captured == {exact_target: 1, neighbor_target: 9}
    assert model.models == before


def test_note_table_predict_searches_multi_position_neighbors(monkeypatch) -> None:
    first = _token(60)
    second = _token(62)
    first_neighbor = _token(61)
    second_neighbor = _token(63)
    target = _token(70)
    model = _model(
        [
            {(first_neighbor, second_neighbor): Counter({target: 10})},
            {(second,): Counter({_token(72): 100})},
        ],
        {
            first: [(first_neighbor, 0.1)],
            second: [(second_neighbor, 0.2)],
        },
    )
    captured = {}

    def choose(population, weights):
        captured.update(zip(population, weights, strict=True))
        return [population[0]]

    monkeypatch.setattr("models.ngram.model.random.choices", choose)

    model.note_table_predict((first, second))

    assert captured == {target: 10}


def test_note_table_predict_stops_after_nearest_context(monkeypatch) -> None:
    context = _token(60)
    near = _token(61)
    far = _token(62)
    near_target = _token(70)
    far_target = _token(71)
    model = _model(
        [{(near,): Counter({near_target: 10}), (far,): Counter({far_target: 10})}],
        {context: [(near, 0.1), (far, 0.2)]},
    )
    captured = {}

    def choose(population, weights):
        captured.update(zip(population, weights, strict=True))
        return [population[0]]

    monkeypatch.setattr("models.ngram.model.random.choices", choose)

    model.note_table_predict((context,))

    assert captured == {near_target: 10}


def test_note_table_predict_backs_off_without_mutating_models(monkeypatch) -> None:
    first = _token(60)
    second = _token(62)
    exact_target = _token(70)
    fallback_target = _token(71)
    models = [
        {(first, second): Counter({exact_target: 1})},
        {(second,): Counter({fallback_target: 10})},
    ]
    model = _model(models)
    before = deepcopy(model.models)
    captured = {}

    def choose(population, weights):
        captured.update(zip(population, weights, strict=True))
        return [population[0]]

    monkeypatch.setattr("models.ngram.model.random.choices", choose)

    model.note_table_predict((first, second))

    assert captured == {fallback_target: 10}
    assert model.models == before
