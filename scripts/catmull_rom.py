"""Centripetal Catmull–Rom for the single-patch representation studies.

Knots depend on distances between ordered interpolation points; alpha=1/2.
Endpoint ghost points are reflected, derived values, not free parameters.
"""
import numpy as np


def catmull_rom_knots(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('At least two finite interpolation points are required')
    increments = np.maximum(np.sqrt(np.linalg.norm(np.diff(points, axis=0), axis=1)), 1e-9)
    knots = np.r_[0., np.cumsum(increments)]
    return knots / knots[-1]


def catmull_rom(points, parameters):
    points = np.asarray(points, dtype=float)
    knots = catmull_rom_knots(points)
    parameters = np.asarray(parameters, dtype=float)
    if parameters.ndim != 1 or not np.isfinite(parameters).all() or np.any((parameters < 0) | (parameters > 1)):
        raise ValueError('Parameters must be a finite vector in [0,1]')
    extended = np.vstack((2*points[0]-points[1], points, 2*points[-1]-points[-2]))
    result = np.empty((len(parameters), points.shape[1]))
    segments = np.clip(np.searchsorted(knots, parameters, side='right')-1, 0, len(points)-2)
    for segment in np.unique(segments):
        selected = segments == segment
        p0, p1, p2, p3 = extended[segment:segment+4]
        h0, h1, h2 = np.maximum(np.sqrt(np.linalg.norm(np.diff(extended[segment:segment+4], axis=0), axis=1)), 1e-9)
        m1 = h1*((p1-p0)/h0-(p2-p0)/(h0+h1)+(p2-p1)/h1)
        m2 = h1*((p2-p1)/h1-(p3-p1)/(h1+h2)+(p3-p2)/h2)
        u = ((parameters[selected]-knots[segment])/(knots[segment+1]-knots[segment]))[:, None]
        result[selected] = ((2*u**3-3*u**2+1)*p1+(u**3-2*u**2+u)*m1
                            +(-2*u**3+3*u**2)*p2+(u**3-u**2)*m2)
    return result
