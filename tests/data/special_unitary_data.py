import itertools
import math
import random
from contextlib import nullcontext as does_not_raise

import pytest

import geomstats.backend as gs
from geomstats.geometry.special_unitary import SpecialUnitaryMatrices
from tests.data_generation import TestData, _LieGroupTestData

CDTYPE = gs.get_default_cdtype()


def sample_matrix(theta, epsilon=1.0, gamma=1.0, mul=1.0):
    return gs.array(
        [[gs.exp(1j * epsilon) * gs.cos(theta), gs.exp(1j * gamma) * gs.sin(theta)],
         [mul * gs.exp(-1j * gamma) * gs.sin(theta), gs.exp(-1j* epsilon) * gs.cos(theta)]],
        dtype=CDTYPE
    )

angle_0 = gs.zeros(3)
angle_close_0 = 1e-10 * gs.array([1.0, -1.0, 1.0])
angle_close_pi_low = (gs.pi - 1e-9) / gs.sqrt(2.0) * gs.array([0.0, 1.0, -1.0])
angle_pi = gs.pi * gs.array([1.0, 0, 0])
angle_close_pi_high = (gs.pi + 1e-9) / gs.sqrt(3.0) * gs.array([-1.0, 1.0, -1])
angle_in_pi_2pi = (gs.pi + 0.3) / gs.sqrt(5.0) * gs.array([-2.0, 1.0, 0.0])
angle_close_2pi_low = (2.0 * gs.pi - 1e-9) / gs.sqrt(6.0) * gs.array([2.0, 1.0, -1])
angle_2pi = 2.0 * gs.pi / gs.sqrt(3.0) * gs.array([1.0, 1.0, -1.0])
angle_close_2pi_high = (2.0 * gs.pi + 1e-9) / gs.sqrt(2.0) * gs.array([1.0, 0.0, -1.0])

elements_all = {
    "angle_0": angle_0,
    "angle_close_0": angle_close_0,
    "angle_close_pi_low": angle_close_pi_low,
    "angle_pi": angle_pi,
    "angle_close_pi_high": angle_close_pi_high,
    "angle_in_pi_2pi": angle_in_pi_2pi,
    "angle_close_2pi_low": angle_close_2pi_low,
    "angle_2pi": angle_2pi,
    "angle_close_2pi_high": angle_close_2pi_high,
}


elements = elements_all
coords = ["extrinsic", "intrinsic"]
orders = ["xyz", "zyx"]
angle_pi_6 = gs.pi / 6.0
cos_angle_pi_6 = gs.cos(angle_pi_6)
sin_angle_pi_6 = gs.sin(angle_pi_6)

cos_angle_pi_12 = gs.cos(angle_pi_6 / 2)
sin_angle_pi_12 = gs.sin(angle_pi_6 / 2)

angles_close_to_pi_all = [
    "angle_close_pi_low",
    "angle_pi",
    "angle_close_pi_high",
]

angles_close_to_pi = angles_close_to_pi_all

class SpecialUnitaryTestData(_LieGroupTestData):
    Space = SpecialUnitaryMatrices

    n_list = random.sample(range(2, 4), 2)
    # space_args_list = list(n_list)
    space_args_list = list(zip(n_list))
    shape_list = [(n, n) for n in n_list] + [(1,), (3,)]
    n_tangent_vecs_list = random.sample(range(2, 10), 4)
    n_points_list = random.sample(range(2, 10), 4)
    n_vecs_list = random.sample(range(2, 10), 4)

    def belongs_test_data(self):
        theta = gs.pi / 3
        smoke_data = [
            dict(n=2, mat=sample_matrix(theta, mul=-1.0), expected=True),
            dict(n=2, mat=sample_matrix(theta, mul=1.0), expected=False),
            dict(n=2, mat=gs.zeros((2, 3), dtype=CDTYPE), expected=False),
            dict(n=3, mat=gs.zeros((2, 3), dtype=CDTYPE), expected=False),
            dict(n=2, mat=gs.zeros((2, 2, 3), dtype=CDTYPE), expected=gs.array([False, False])),
            dict(
                n=2,
                mat=gs.stack(
                    [
                        sample_matrix(theta / 2, mul=-1.0),
                        sample_matrix(theta / 2, mul=1.0),
                    ]
                ),
                expected=gs.array([True, False]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def dim_test_data(self):
        smoke_data = [
            dict(n=2, expected=1),
            dict(n=3, expected=3),
            dict(n=4, expected=6),
        ]
        return self.generate_tests(smoke_data)

    def identity_test_data(self):
        smoke_data = [
            dict(n=2, expected=gs.eye(2, dtype=CDTYPE)),
            dict(n=3, expected=gs.eye(3, dtype=CDTYPE)),
            dict(n=4, expected=gs.eye(4, dtype=CDTYPE)),
        ]
        return self.generate_tests(smoke_data)

    def is_tangent_test_data(self):
        point = SpecialUnitaryMatrices(2).random_uniform()
        vec_1 = gs.array([[0.0 + 1j * 0.0, 2 + 1j], [-2 + 1j, 0.0 + 1j * 0.0]])
        vec_2 = gs.array([[0.0 + 1j * 0.0, 2 + 1j], [-2 + 1j, 1.0 + 1j]])
        smoke_data = [
            dict(n=2, vec=vec_1, base_point=None, expected=True),
            dict(n=2, vec=vec_2, base_point=None, expected=False),
            dict(n=2, vec=[vec_1, vec_2], base_point=None, expected=[True, False]),
            dict(
                n=2,
                vec=SpecialUnitaryMatrices(2).compose(point, vec_1),
                base_point=point,
                expected=True,
            ),
            dict(
                n=2,
                vec=SpecialUnitaryMatrices(2).compose(point, vec_2),
                base_point=point,
                expected=False,
            ),
        ]
        return self.generate_tests(smoke_data)
    #
    # def is_tangent_compose_test_data(self):
    #     """
    #     TODO
    #     """
    #     point = SpecialUnitaryMatrices(2).random_uniform()
    #     theta = 1.0
    #     vec_1 = gs.array([[0.0, -theta], [theta, 0.0]])
    #     vec_2 = gs.array([[0.0, -theta], [theta, 1.0]])
    #
    #     smoke_data = [
    #         dict(
    #             n=2,
    #             vec=SpecialUnitaryMatrices(2).compose(point, vec_1),
    #             point=point,
    #             expected=True,
    #         ),
    #         dict(
    #             n=2,
    #             vec=SpecialUnitaryMatrices(2).compose(point, vec_2),
    #             point=point,
    #             expected=False,
    #         ),
    #         dict(
    #             n=2,
    #             vec=[
    #                 SpecialUnitaryMatrices(2).compose(point, vec_1),
    #                 SpecialUnitaryMatrices(2).compose(point, vec_2),
    #             ],
    #             point=point,
    #             expected=[True, False],
    #         ),
    #     ]
    #     return self.generate_tests(smoke_data)
    #
    # def to_tangent_test_data(self):
    #     """
    #     TODO
    #     """
    #     theta = 1.0
    #     smoke_data = [
    #         dict(
    #             n=2,
    #             vec=[[0.0, -theta], [theta, 0.0]],
    #             base_point=None,
    #             expected=[[0.0, -theta], [theta, 0.0]],
    #         ),
    #         dict(
    #             n=2,
    #             vec=[[1.0, -math.pi], [math.pi, 1.0]],
    #             base_point=[
    #                 [gs.cos(math.pi), -1 * gs.sin(math.pi)],
    #                 [gs.sin(math.pi), gs.cos(math.pi)],
    #             ],
    #             expected=[[0.0, -math.pi], [math.pi, 0.0]],
    #         ),
    #     ]
    #     return self.generate_tests(smoke_data)
    #
    # def distance_broadcast_shape_test_data(self):
    #     """
    #     TODO
    #     """
    #     n_list = [2, 3]
    #     n_samples_list = random.sample(range(1, 20), 2)
    #     smoke_data = [
    #         dict(n=n, n_samples=n_samples)
    #         for n, n_samples in zip(n_list, n_samples_list)
    #     ]
    #     return self.generate_tests(smoke_data)
    #
    def projection_test_data(self):
        n_list = [2, 3]
        smoke_data = [
            dict(
                n=n,
                point_type="vector",
                mat=gs.eye(n, dtype=CDTYPE) + 1e-12 * gs.ones((n, n), dtype=CDTYPE),
                expected=gs.eye(n, dtype=CDTYPE),
            )
            for n in n_list
        ]
        return self.generate_tests(smoke_data)

    def projection_shape_test_data(self):

        n_list = [2, 3]
        n_samples_list = random.sample(range(2, 20), 2)
        random_data = [
            dict(
                n=n,
                n_samples=n_samples,
                expected=(n_samples, n, n),
            )
            for n, n_samples in zip(n_list, n_samples_list)
        ]
        return self.generate_tests([], random_data)

    def compose_with_inverse_is_identity_test_data(self):
        smoke_data = []
        for space_args in list(zip(self.n_list)):
            smoke_data += [dict(space_args=space_args)]
        return self.generate_tests(smoke_data)

    def compose_test_data(self):
        smoke_data = [
            dict(
                n=2,
                point_a=gs.array(
                                [[gs.exp(1j), 0.0],
                                [0.0, gs.exp(-1j)]],
                                dtype=CDTYPE
                                ),
                point_b = gs.array(
                                [[0.0, gs.exp(1j)],
                                [gs.exp(-1j), 0.0]],
                                dtype=CDTYPE
                                ),
                expected=gs.array(
                                [[0.0, gs.exp(2j)],
                                [gs.exp(-2j), 0.0]],
                                dtype=CDTYPE
                                ),
            )
        ]
        return self.generate_tests(smoke_data)

    def log_test_data(self):
        smoke_data = [
            dict(
                n=2,
                point=gs.array([[gs.exp(1j), 0.0],
                                      [0.0, gs.exp(-1j)]]),
                base_point=None,
                expected=gs.array([[1j, 0.0],
                                      [0.0, -1j]]),
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_test_data(self):
        smoke_data = [
            dict(
                n=2,
                tangent_vec=gs.array([[gs.exp(1j), 0.0],
                                      [0.0, gs.exp(-1j)]]),
                base_point=None,
                expected=gs.array([[gs.exp(gs.exp(1j)), 0.0],
                                      [0.0, gs.exp(gs.exp(-1j))]],
            )
        )
        ]
        return self.generate_tests(smoke_data)
    
    def compose_shape_test_data(self):
        smoke_data = [
            dict(n=2, n_samples=4),
            dict(n=3, n_samples=4),
            dict(n=4, n_samples=4),
        ]
        return self.generate_tests(smoke_data)
