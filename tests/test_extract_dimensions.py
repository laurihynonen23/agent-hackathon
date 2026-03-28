from app.extract_dimensions import parse_dimension_text


def test_parse_dimension_text_scale():
    kind, value, meta = parse_dimension_text("1 : 50")
    assert kind == "drawing_scale"
    assert value == 50.0
    assert meta["scale_denominator"] == 50.0


def test_parse_dimension_text_mm_value():
    kind, value, meta = parse_dimension_text("14472")
    assert kind == "dimension_mm"
    assert value == 14.472
    assert meta["raw_value_mm"] == 14472.0


def test_parse_dimension_text_material_size():
    kind, value, meta = parse_dimension_text("28x170")
    assert kind == "material_size"
    assert value == 0.170
    assert meta["cover_mm"] == 170.0

