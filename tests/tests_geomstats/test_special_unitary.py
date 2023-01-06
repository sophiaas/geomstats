import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.special_unitary import SpecialUnitaryMatrices
from tests.conftest import Parametrizer, TestCase, pytorch_backend
from tests.data.special_unitary_data import SpecialUnitaryTestData
from tests.geometry_test_cases import LieGroupTestCase

EPSILON = 1e-5


class TestSpecialUnitary(LieGroupTestCase, metaclass=Parametrizer):

    skip_test_exp_after_log = True
    skip_test_log_after_exp = True
    skip_test_projection_belongs = True
    skip_test_random_tangent_vec_is_tangent = True
    skip_test_to_tangent_at_identity_belongs_to_lie_algebra = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = SpecialUnitaryTestData()

    def test_belongs(self, n, mat, expected):
        self.assertAllClose(self.Space(n).belongs(gs.array(mat)), gs.array(expected))

    def test_dim(self, n, expected):
        self.assertAllClose(self.Space(n).dim, expected)

    def test_identity(self, n, expected):
        self.assertAllClose(self.Space(n).identity, gs.array(expected))

    def test_is_tangent(self, n, vec, base_point, expected):
        group = self.Space(n)
        self.assertAllClose(
            group.is_tangent(gs.array(vec), base_point), gs.array(expected)
        )

    def test_projection(self, n, mat, expected):
        group = self.Space(n=n)
        self.assertAllClose(group.projection(mat), expected)

    def test_projection_shape(self, n, n_samples, expected):
        group = self.Space(n=n)
        self.assertAllClose(
            gs.shape(group.projection(group.random_point(n_samples))), expected
        )

    def test_compose_with_inverse_is_identity(self, space_args):
        group = SpecialUnitaryMatrices(*space_args)
        point = gs.squeeze(group.random_point())
        inv_point = group.inverse(point)
        self.assertAllClose(group.compose(point, inv_point), group.identity)

    def test_compose(self, n, point_a, point_b, expected):
        group = SpecialUnitaryMatrices(n)
        result = group.compose(point_a, point_b)
        self.assertAllClose(result, expected)

    def test_exp(self, n, tangent_vec, base_point, expected):
        group = self.Space(n)
        result = group.exp(tangent_vec, base_point)
        self.assertAllClose(result, expected)
    
    def test_log(self, n, point, base_point, expected):
        group = self.Space(n)
        result = group.log(point=point, base_point=base_point)
        self.assertAllClose(result, expected)

    def test_compose_shape(self, n, n_samples):
        group = self.Space(n)
        n_points_a = group.random_uniform(n_samples=n_samples)
        n_points_b = group.random_uniform(n_samples=n_samples)
        one_point = group.random_uniform(n_samples=1)

        result = group.compose(one_point, n_points_a)
        self.assertAllClose(gs.shape(result), (n_samples,) + group.shape)

        result = group.compose(n_points_a, one_point)
        self.assertAllClose(gs.shape(result), (n_samples,) + group.shape)

        result = group.compose(n_points_a, n_points_b)
        self.assertAllClose(gs.shape(result), (n_samples,) + group.shape)
