"""Detector-network geometry: positions, antenna patterns, delays, coherent projection.

Frames. Everything is Earth-fixed (ECEF: x through lat 0 / lon 0, z through the
north pole). G2Net does not give GPS times, so a sky position is an Earth-fixed
direction n^ = (cos(lat) cos(lon), cos(lat) sin(lon), sin(lat)) pointing from
the Earth's centre to the source. (In LAL terms: lon = ra - gmst, lat = dec.)

Detector k: vertex r_k (WGS-84 geodetic -> ECEF) and arm unit vectors x_k, y_k.
Its response tensor is D_k = (x_k x_k^T - y_k y_k^T) / 2.

A plane wave from n^ travels along -n^. It reaches r_k at
    t_k = t_geocentre - (r_k . n^) / c,
so the delay relative to the geocentre is dt_k(n^) = -(r_k . n^) / c.

Polarisation basis (LAL convention, polarisation angle psi): X, Y orthonormal
and orthogonal to n^. e+ = X X^T - Y Y^T, ex = X Y^T + Y X^T.
Antenna patterns: F+_k = D_k : e+, Fx_k = D_k : ex (double contraction).
Detector strain: h_k(t) = F+_k h+(t - dt_k) + Fx_k hx(t - dt_k).

Coherent projection. Let z_k be the complex SNR of detector k with a template
normalised in that detector, and sigma_k = sqrt(<h|h>_k) the template norm at a
reference distance (the detector's sensitivity to that template). For a signal
the vector (z_k / ...) lies in the span of the two real vectors
w+ = (sigma_k F+_k)_k and wx = (sigma_k Fx_k)_k. With W = [w+, wx] (3 x 2):
    P = W (W^T W)^{-1} W^T        (projector, rank 2, independent of psi)
    rho_coh^2  = z^H P z           (noise: chi2 with 4 dof)
    rho_null^2 = z^H (I - P) z     (noise: chi2 with 2 dof; signal: ~0)
Harry & Fairhurst, PRD 83, 084002 (2011).
"""

import numpy as np

C_SI = 299792458.0
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563

# Vertex latitude, longitude [rad], elevation [m], x/y arm azimuths [rad, clockwise
# (east) of north] and arm altitudes [rad]. Values of LAL's cached frDetector.
DETECTOR_SITES = {
    "H1": (0.81079526383, -2.08405676917, 142.554, 5.654877185821533, 4.084080696105957,
           -0.0006195000023581088, 1.249999968422344e-05),
    "L1": (0.53342313506, -1.58430937078, -6.574, 4.403177738189697, 2.8323814868927,
           -0.00031209998996928334, -0.000610699993558228),
    "V1": (0.76151183984, 0.18333805213, 51.884, 0.3391628563404083, 5.051551818847656, 0.0, 0.0),
}


def _local_frame(lat, lon):
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    return east, north, up


def detector_geometry(name: str):
    """(position r [m], response tensor D) of a detector in ECEF."""
    lat, lon, h, az_x, az_y, alt_x, alt_y = DETECTOR_SITES[name]
    e2 = WGS84_F * (2 - WGS84_F)
    nrad = WGS84_A / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    r = np.array([(nrad + h) * np.cos(lat) * np.cos(lon),
                  (nrad + h) * np.cos(lat) * np.sin(lon),
                  (nrad * (1 - e2) + h) * np.sin(lat)])
    east, north, up = _local_frame(lat, lon)

    def arm(az, alt):
        return np.cos(alt) * (np.sin(az) * east + np.cos(az) * north) + np.sin(alt) * up

    x, y = arm(az_x, alt_x), arm(az_y, alt_y)
    return r, 0.5 * (np.outer(x, x) - np.outer(y, y))


def sky_direction(lon, lat):
    lon, lat = np.asarray(lon), np.asarray(lat)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def polarisation_basis(lon, lat, psi):
    """X, Y of LAL's DetResponse with gha = -lon (gmst = 0, ra = lon), shape (..., 3)."""
    gha = -np.asarray(lon)
    sd, cd = np.sin(lat), np.cos(lat)
    sg, cg = np.sin(gha), np.cos(gha)
    sp, cp = np.sin(psi), np.cos(psi)
    X = np.stack([-cp * sg - sp * cg * sd, -cp * cg + sp * sg * sd, sp * cd * np.ones_like(sg)], axis=-1)
    Y = np.stack([sp * sg - cp * cg * sd, sp * cg + cp * sg * sd, cp * cd * np.ones_like(sg)], axis=-1)
    return X, Y


def antenna_patterns(D, lon, lat, psi=0.0):
    """F+, Fx of response tensor D for Earth-fixed sky positions."""
    X, Y = polarisation_basis(lon, lat, psi)
    fp = np.einsum("...i,ij,...j->...", X, D, X) - np.einsum("...i,ij,...j->...", Y, D, Y)
    fx = 2.0 * np.einsum("...i,ij,...j->...", X, D, Y)
    return fp, fx


def geocentre_delay(r, lon, lat):
    """Arrival time at position r minus arrival time at the geocentre [s]."""
    return -sky_direction(lon, lat) @ r / C_SI


def fibonacci_sky(n: int):
    """n nearly uniform Earth-fixed directions (lon, lat) on the sphere."""
    i = np.arange(n) + 0.5
    lat = np.arcsin(1 - 2 * i / n)
    lon = np.mod(np.pi * (1 + 5 ** 0.5) * i, 2 * np.pi) - np.pi
    return lon, lat


class Network:
    """Geometry of an ordered detector list on a fixed sky grid."""

    def __init__(self, detectors=("H1", "L1", "V1"), n_sky: int = 2000):
        self.detectors = detectors
        geo = [detector_geometry(d) for d in detectors]
        self.positions = np.stack([g[0] for g in geo])
        self.tensors = np.stack([g[1] for g in geo])
        self.lon, self.lat = fibonacci_sky(n_sky)
        # (S, K) delays and antenna patterns at psi = 0
        self.delays = np.stack([geocentre_delay(r, self.lon, self.lat) for r in self.positions], axis=-1)
        f = [antenna_patterns(D, self.lon, self.lat) for D in self.tensors]
        self.fplus = np.stack([a for a, _ in f], axis=-1)
        self.fcross = np.stack([b for _, b in f], axis=-1)

    def max_pair_delay(self, i: int, j: int) -> float:
        return np.linalg.norm(self.positions[i] - self.positions[j]) / C_SI

    def projectors(self, sigma: np.ndarray) -> np.ndarray:
        """Coherent projectors P (S, K, K) for detector sensitivities sigma (K,)."""
        W = np.stack([self.fplus * sigma, self.fcross * sigma], axis=-1)  # (S, K, 2)
        G = np.einsum("ska,skb->sab", W, W)
        return np.einsum("ska,sab,slb->skl", W, np.linalg.inv(G), W)
