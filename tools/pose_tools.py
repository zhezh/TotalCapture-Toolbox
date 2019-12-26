import numpy as np
from pyquaternion import Quaternion


def get_linear_accel(accel, orient, g=-9.83):
    """
    accel_measurement(in b frame) f^b = R_{bn}(a^n - g^n)
    we derive global linear accel: a^n = R_{nb}f^b + g^n
    orient represents R_{nb}
    here, n frame equals inertial frame
    :param accel: accel with gravity, aka f^b
    :param orient: type Quaternion; global orientation in quaternion
    :param g: gravity accel
    :return: linear accel in inertial frame, aka a^n
    """
    # assert type(orient) == type(Quaternion([0., 1., 0., 0.])), 'wrong orientation type'
    assert isinstance(orient, Quaternion), 'wrong orientation type'
    g_n = np.array([0., 0., g])
    # linear_accel = orient.inverse.rotate(accel) + g_n
    linear_accel = orient.rotate(accel) + g_n
    return linear_accel


def center_accel(frame1, frame2, frame3, unit='inch', tau=0.0166666):
    """

    :param frame1:
    :param frame2:
    :param frame3:
    :param unit:
    :param tau:
    :return: unit is m/(s^2)
    """
    assert unit in ('inch', 'm'), 'wrong unit'
    if isinstance(frame1, (list, tuple)):
        frame1 = np.array(frame1, dtype=np.float)
    if isinstance(frame2, (list, tuple)):
        frame2 = np.array(frame2, dtype=np.float)
    if isinstance(frame3, (list, tuple)):
        frame3 = np.array(frame3, dtype=np.float)

    acc = (frame3 - 2*frame2 + frame1)/(tau*tau)
    if unit == 'inch':
        acc = acc*0.0254
    return acc
