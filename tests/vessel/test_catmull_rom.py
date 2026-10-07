import numpy as np
from scripts.catmull_rom import catmull_rom, catmull_rom_knots


def test_nonuniform_points_are_interpolated_with_continuous_tangents():
    points = np.array([[0.,0,0],[.2,1.5,0],[2.5,.4,.1],[2.6,2,.5],[4,0,0]])
    knots = catmull_rom_knots(points)
    assert np.allclose(catmull_rom(points, knots), points, atol=1e-10)
    epsilon = 1e-7
    for t in knots[1:-1]:
        value = catmull_rom(points, np.array([t]))[0]
        left = (value-catmull_rom(points, np.array([t-epsilon]))[0])/epsilon
        right = (catmull_rom(points, np.array([t+epsilon]))[0]-value)/epsilon
        assert np.allclose(left, right, atol=5e-4)


def test_moved_interpolation_point_and_fixed_endpoints():
    points = np.array([[0.,0,0],[1,1,0],[2,-1,0],[3,0,0]])
    points[1] += [0,.5,.2]
    curve = catmull_rom(points,catmull_rom_knots(points))
    assert np.allclose(curve,points)


def test_repeated_points_remain_finite():
    points = np.array([[0.,0,0],[1,1,0],[1,1,0],[2,-1,0],[3,0,0]])
    assert np.isfinite(catmull_rom(points,np.linspace(0,1,801))).all()
    assert np.allclose(catmull_rom(points,catmull_rom_knots(points)),points)
