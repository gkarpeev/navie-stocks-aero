"""Text the figures and slides print: plural forms, digits of a value with its uncertainty."""

import pytest

from lab02.plots import EDGES, VERTICES, _count, _fmt, _sci
from lab02.report import markdown_table


@pytest.mark.parametrize(
    "n, text",
    [
        (1, "1 ребро"),
        (2, "2 ребра"),
        (4, "4 ребра"),
        (5, "5 рёбер"),
        (11, "11 рёбер"),
        (12, "12 рёбер"),
        (14, "14 рёбер"),
        (19, "19 рёбер"),
        (21, "21 ребро"),
        (22, "22 ребра"),
        (111, "111 рёбер"),
    ],
)
def test_russian_plural_of_edges(n, text):
    assert _count(n, EDGES) == text


def test_russian_plural_of_vertices():
    assert [_count(n, VERTICES) for n in (1, 3, 9, 20)] == [
        "1 вершина",
        "3 вершины",
        "9 вершин",
        "20 вершин",
    ]


def test_value_keeps_the_digits_its_uncertainty_supports():
    assert _fmt(0.79370, 0.01328) == "0.794 ± 0.013"
    assert _fmt(-4.3638, 6.427) == "-4.4 ± 6.4"
    assert _fmt(1787.3, 25.0) == "1787 ± 25"


def test_scientific_form_only_outside_hundredths_to_hundreds():
    assert _sci(0.234) == "0.234"
    assert _sci(1538.17) == r"1.5\cdot10^{3}"
    assert _sci(0.00031) == r"3.1\cdot10^{-4}"


def test_markdown_table_has_a_separator_per_column():
    table = markdown_table(["a", "b"], [[1, 2], [3, 4]])
    assert table.splitlines() == ["| a | b |", "|---|---|", "| 1 | 2 |", "| 3 | 4 |"]
