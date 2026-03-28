from app.geometry import build_rectangular_footprint, gable_wall_area


def test_rectangular_footprint_perimeter():
    polygon = build_rectangular_footprint(9.168, 14.472)
    assert round(polygon.length, 3) == round(2 * (9.168 + 14.472), 3)


def test_gable_wall_area():
    area = gable_wall_area(width_m=9.168, eave_height_m=4.015, total_height_m=6.9)
    assert round(area, 3) == round(9.168 * 4.015 + 0.5 * 9.168 * (6.9 - 4.015), 3)

