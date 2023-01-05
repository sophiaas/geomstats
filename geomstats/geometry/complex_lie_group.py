"""Complex Lie groups.

Lead author: Sophia Sanborn.
"""

import abc

import geomstats.backend as gs
from geomstats.geometry.complex_manifold import ComplexManifold
from geomstats.geometry.complex_matrices import ComplexMatrices

ATOL = 1e-6
CDTYPE = gs.get_default_cdtype()


class ComplexMatrixLieGroup(ComplexManifold, abc.ABC):
    """Class for complex matrix Lie groups."""

    def __init__(self, dim, representation_dim, lie_algebra=None, **kwargs):
        super().__init__(
            dim=dim, shape=(representation_dim, representation_dim), **kwargs
        )
        self.lie_algebra = lie_algebra
        self.representation_dim = representation_dim

    @property
    def identity(self):
        """Return the identity."""
        return gs.eye(self.representation_dim, dtype=CDTYPE)

    @staticmethod
    def compose(point_a, point_b):
        """Perform function composition corresponding to the Lie group.

        Multiply the elements `point_a` and `point_b`.

        Parameters
        ----------
        point_a : array-like, shape=[..., {dim, [n, n]}]
            Left factor in the product.
        point_b : array-like, shape=[..., {dim, [n, n]}]
            Right factor in the product.

        Returns
        -------
        composed : array-like, shape=[..., {dim, [n, n]}]
            Product of point_a and point_b along the first dimension.
        """
        return ComplexMatrices.mul(point_a, point_b)

    @classmethod
    def inverse(cls, point):
        """Compute the inverse law of the Lie group.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n, n]}]
            Point to be inverted.

        Returns
        -------
        inverse : array-like, shape=[..., {dim, [n, n]}]
            Inverted point.
        """
        return gs.linalg.inv(point)

    def tangent_translation_map(self, point, left=True, inverse=False):
        r"""Compute the push-forward map by the left/right translation.

        Compute the push-forward map, of the left/right translation by the
        point. It corresponds to the tangent map, or differential of the
        group multiplication by the point or its inverse. For groups with a
        vector representation, it is only implemented at identity, but it can
        be used at other points by passing `inverse=True`. This method wraps
        the jacobian translation which actually computes the matrix
        representation of the map.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n, n]]
            Point.
        left : bool
            Whether to calculate the differential of left or right
            translations.
            Optional, default: True
        inverse : bool,
            Whether to inverse the jacobian matrix. If True, the push forward
            by the translation by the inverse of point is returned.
            Optional, default: False.

        Returns
        -------
        tangent_map : callable
            Tangent map of the left/right translation by point. It can be
            applied to tangent vectors.
        """
        if inverse:
            point = self.inverse(point)
        if left:
            return lambda tangent_vec: self.compose(point, tangent_vec)
        return lambda tangent_vec: self.compose(tangent_vec, point)

    def lie_bracket(self, tangent_vector_a, tangent_vector_b, base_point=None):
        """Compute the lie bracket of two tangent vectors.

        For matrix Lie groups with tangent vectors A,B at the same base point P
        this is given by (translate to identity, compute commutator, go back)
        :math:`[A,B] = A_P^{-1}B - B_P^{-1}A`

        Parameters
        ----------
        tangent_vector_a : array-like, shape=[..., n, n]
            Tangent vector at base point.
        tangent_vector_b : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        bracket : array-like, shape=[..., n, n]
            Lie bracket.
        """
        if base_point is None:
            base_point = self.identity
        inverse_base_point = self.inverse(base_point)

        first_term = ComplexMatrices.mul(inverse_base_point, tangent_vector_b)
        first_term = ComplexMatrices.mul(tangent_vector_a, first_term)

        second_term = ComplexMatrices.mul(inverse_base_point, tangent_vector_a)
        second_term = ComplexMatrices.mul(tangent_vector_b, second_term)

        return first_term - second_term

    def is_tangent(self, vector, base_point=None, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim_embedding]
            Vector.
        base_point : array-like, shape=[..., dim_embedding]
            Point in the Lie group.
            Optional. default: identity.
        atol : float
            Precision at which to evaluate if the rotation part is
            skew-symmetric.
            Optional. default: 1e-6

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        if base_point is None:
            tangent_vec_at_id = vector
        else:
            tangent_vec_at_id = self.compose(self.inverse(base_point), vector)
        return self.lie_algebra.belongs(tangent_vec_at_id, atol)

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
        regularized = self.lie_algebra.projection(tangent_vec_at_id)
        return self.compose(base_point, regularized)

    @classmethod
    def exp(cls, tangent_vec, base_point=None):
        r"""
        Exponentiate a left-invariant vector field from a base point.

        The vector input is not an element of the Lie algebra, but of the
        tangent space at base_point: if :math:`g` denotes `base_point`,
        :math:`v` the tangent vector, and :math:`V = g^{-1} v` the associated
        Lie algebra vector, then

        .. math::

            \exp(v, g) = mul(g, \exp(V))

        Therefore, the Lie exponential is obtained when base_point is None, or
        the identity.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.
            Optional, defaults to identity if None.

        Returns
        -------
        point : array-like, shape=[..., n, n]
            Left multiplication of `exp(algebra_mat)` with `base_point`.
        """
        expm = gs.linalg.expm
        if base_point is None:
            return expm(tangent_vec)
        lie_algebra_vec = cls.compose(cls.inverse(base_point), tangent_vec)
        return cls.compose(base_point, cls.exp(lie_algebra_vec))

    @classmethod
    def log(cls, point, base_point=None):
        r"""
        Compute a left-invariant vector field bringing base_point to point.

        The output is a vector of the tangent space at base_point, so not a Lie
        algebra element if it is not the identity.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point.
        base_point : array-like, shape=[..., n, n]
            Base point.
            Optional, defaults to identity if None.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n, n]
            Matrix such that `exp(tangent_vec, base_point) = point`.

        Notes
        -----
        Denoting `point` by :math:`g` and `base_point` by :math:`h`,
        the output satisfies:

        .. math::

            g = \exp(\log(g, h), h)
        """
        logm = gs.linalg.logm
        if base_point is None:
            return logm(point)
        lie_algebra_vec = logm(cls.compose(cls.inverse(base_point), point))
        return cls.compose(base_point, lie_algebra_vec)