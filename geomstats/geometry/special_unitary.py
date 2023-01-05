"""The Special Unitary Group SU(n).

Lead author: Sophia Sanborn.
"""

import geomstats.backend as gs
from geomstats.geometry.base import LevelSet
from geomstats.geometry.complex_matrices import ComplexMatrices
from geomstats.geometry.hermitian_matrices import HermitianMatrices
from geomstats.geometry.complex_lie_group import ComplexMatrixLieGroup
from geomstats.geometry.skew_hermitian_matrices import SkewHermitianMatrices
from geomstats.geometry.complex_general_linear import ComplexGeneralLinear

CDTYPE = gs.get_default_cdtype()

class SpecialUnitaryMatrices(ComplexMatrixLieGroup, LevelSet):
    """Class for special unitary groups in matrix representation.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n, **kwargs):

        self.n = n
        self.value = gs.eye(n, dtype=CDTYPE)

        super().__init__(
            dim=int((n * (n - 1)) / 2),
            representation_dim=n,
            lie_algebra=SkewHermitianMatrices(n=n),
            default_coords_type="extrinsic",
            **kwargs,
        )

    def _define_embedding_space(self):
        return ComplexGeneralLinear(self.n, positive_det=True)

    def _aux_submersion(self, point):
        # lambda x: ComplexMatrices.mul(ComplexMatrices.transconjugate(x), x)
        return ComplexMatrices.mul(ComplexMatrices.transconjugate(point), point)

    def submersion(self, point):
        """Submersion that defines the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]

        Returns
        -------
        submersed_point : array-like, shape=[..., n. n]
        """
        return self._aux_submersion(point) - self._value

    def tangent_submersion(self, vector, point):
        """Tangent submersion.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
        point : array-like, shape=[..., n, n]

        Returns
        -------
        submersed_vector : array-like, shape=[..., n, n]
        """
        # tangent_submersion=lambda v, x: 2 * ComplexMatrices.to_symmetric(ComplexMatrices.mul(ComplexMatrices.transconjugate(x), v)),
        return 2 * ComplexMatrices.to_symmetric(ComplexMatrices.mul(ComplexMatrices.transconjugate(point), vector))

    @classmethod
    def inverse(cls, point):
        """Return the conjugate transpose of point.
        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point in SU(n).
        Returns
        -------
        inverse : array-like, shape=[..., n, n]
            Inverse.
        """
        return ComplexMatrices.transconjugate(point)

    def to_tangent(self, vector, base_point=None):
        """Project a vector onto the tangent space at a base point.
        Parameters
        ----------
        vector : array-like, shape=[..., {dim, [n, n]}]
            Vector to project. Its shape must match the shape of base_point.
        base_point : array-like, shape=[..., {dim, [n, n]}], optional
            Point of the group.
            Optional, default: identity.
        Returns
        -------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        """
        if base_point is None:
            return self.lie_algebra.projection(vector)
        tangent_vec_at_id = self.compose(self.inverse(base_point), vector)

        regularized = ComplexMatrices.to_skew_hermitian(tangent_vec_at_id)
        return self.compose(base_point, regularized)

    def projection(self, point):
        """Project a matrix on SU(n) by minimizing the Frobenius norm.
        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix.
        Returns
        -------
        rot_mat : array-like, shape=[..., n, n]
            Rotation matrix.
        """
        aux_mat = self.submersion(point)
        inv_sqrt_mat = HermitianMatrices.powerm(aux_mat, -1 / 2)
        rotation_mat = ComplexMatrices.mul(point, inv_sqrt_mat)
        det = gs.linalg.det(rotation_mat)
        return rotation_mat

    def random_point(self, n_samples=1, bound=1.0):
        """Sample in SU(n) from the uniform distribution.
        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Unused.
        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Points sampled on the SO(n).
        """
        return self.random_uniform(n_samples)

    def random_uniform(self, n_samples=1):
        """Sample in SU(n) from the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        tol : unused
        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Points sampled on the SO(n).
        """
        if n_samples == 1:
            size = (self.n, self.n)
        else:
            size = (n_samples, self.n, self.n)
        ginibre_ensemble = gs.random.normal(size=size, scale=1/2) + 1j * gs.random.normal(size=size, scale=1/2)
        rotation_mat, _ = gs.linalg.qr(ginibre_ensemble)
        det = gs.linalg.det(rotation_mat)
        if len(rotation_mat.shape) > 2:
            det = gs.reshape(det, (det.shape[0], 1, 1))
        det_scaled = det ** (1 / self.n)
        SU_mat = rotation_mat / det_scaled
        return SU_mat