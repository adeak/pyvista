import math
from hypothesis import assume, given
from hypothesis.strategies import composite, iterables, tuples, lists, floats, integers, sampled_from, one_of
import numpy as np
import pytest
import vtk

import pyvista
from pyvista.plotting import system_supports_plotting
# TODO: do we need OFF_SCREEN stuff? And as per tests/test_renderer.py or tests/plotting/test_plotting.py?
# TODO: invalid cases, once checks are in place


@pytest.fixture()
def light():
    return pyvista.Light()


@composite
def numeric_triple(draw, min_value=None, max_value=None, allow_infinity=None, allow_nan=False):
    """Return a tuple of length 3, containing numbers, int or float"""
    _floats = floats(min_value=min_value, max_value=max_value, allow_infinity=allow_infinity, allow_nan=allow_nan)
    _integers = integers(min_value=min_value, max_value=max_value)
    return tuple(draw(lists(one_of(_floats, _integers), min_size=3, max_size=3)))


@given(position=numeric_triple(allow_nan=False, allow_infinity=False),
       color=numeric_triple(min_value=0.0, max_value=1.0))
def test_init(position, color):
    position = position
    color = color
    light_type = 'headlight'
    light = pyvista.Light(position=position, color=color, light_type=light_type)
    assert isinstance(light, pyvista.Light)
    assert light.position == pytest.approx(position)
    assert light.ambient_color == pytest.approx(color)
    assert light.diffuse_color == pytest.approx(color)
    assert light.specular_color == pytest.approx(color)
    assert light.light_type == pyvista.Light.HEADLIGHT

    # check repr too
    assert repr(light) is not None


def test_colors(light):
    color = (0, 1, 0)
    light.diffuse_color = color
    assert light.diffuse_color == color
    color = (0, 0, 1)
    light.specular_color = color
    assert light.specular_color == color
    color = (1, 0, 0)
    light.ambient_color = color
    assert light.ambient_color == color

    old_color, color = color, (1, 1, 0)
    light.set_color(color)
    assert light.diffuse_color == light.specular_color == color
    assert light.ambient_color == old_color


def test_positioning(light):
    position = (1, 1, 1)
    light.position = position
    assert light.position == position
    # with no transformation matrix this is also the world position
    assert light.world_position == position

    focal_point = (2, 2, 2)
    light.focal_point = focal_point
    assert light.focal_point == focal_point
    # with no transformation matrix this is also the world focal point
    assert light.world_focal_point == focal_point

    elev, azim = (45, 30)
    expected_position = (0.5 / math.sqrt(2),
                         1 / math.sqrt(2),
                         math.sqrt(3) / (2 * math.sqrt(2)))  # TODO: fix this style
    light.positional = True
    light.set_direction_angle(elev, azim)
    assert not light.positional
    assert light.focal_point == (0, 0, 0)
    assert all(math.isclose(coord_have, coord_expect) for coord_have, coord_expect
               in zip(light.position, expected_position))  # TODO: fix this style


@given(intensity=floats(min_value=0.0, max_value=1.0))
def test_intensity_should_accept_0_to_1(intensity, light):
    light.intensity = intensity
    assert light.intensity == pytest.approx(intensity)


@given(intensity=floats(allow_nan=False))
def test_intensity_should_clamp_out_of_range(intensity, light):
    assume(not (0.0 < intensity < 1.0))
    light.intensity = intensity
    assert light.intensity == pytest.approx(np.clip(intensity, 0.0, 1.0))


def test_switch_state(light):
    light.switch_on()
    assert light.is_on
    light.switch_off()
    assert not light.is_on
    light.switch()
    assert light.is_on
    light.is_on = False
    assert not light.is_on


def test_positional(light):
    # default is directional light
    assert not light.positional
    light.positional_on()
    assert light.positional
    light.positional_off()
    assert not light.positional
    light.positional = True
    assert light.positional


def test_shape():
    exponent = 1.5
    light.exponent = exponent
    assert light.exponent == exponent

    attenuation_values = (3, 2, 1)
    light.attenuation_values = attenuation_values
    assert light.attenuation_values == attenuation_values


# TODO, is this the correct range?
@given(value=floats(min_value=0.0, max_value=1.0))
def test_shadow_attenuation_should_accept_0_to_1(value, light):
    light.shadow_attenuation = value
    assert light.shadow_attenuation == pytest.approx(value)


@given(angle=one_of(integers(min_value=0, max_value=360), floats(min_value=0, max_value=360)))
def test_cone_angle_should_accept_0_to_360(angle, light):
    light.cone_angle = angle
    assert light.cone_angle == pytest.approx(angle)


@given(enum_code=sampled_from(pyvista.lights.LightType))
def test_light_type_should_accept_int_or_enum(enum_code, light):
    int_code = int(enum_code)
    # test that both codes work
    light.light_type = int_code
    assert light.light_type == int_code
    light.light_type = enum_code
    assert light.light_type == enum_code


def test_type_setters(light):
    light.set_headlight()
    assert light.is_headlight
    light.set_camera_light()
    assert light.is_camera_light
    light.set_scene_light()
    assert light.is_scene_light


def test_type_invalid():
    with pytest.raises(TypeError):
        light = pyvista.Light(light_type=['invalid'])
    with pytest.raises(ValueError):
        light = pyvista.Light(light_type='invalid')

    light = pyvista.Light()

    with pytest.raises(TypeError):
        light.light_type = ['invalid']


def test_from_vtk():
    vtk_light = vtk.vtkLight()

    # pyvista attr -- value -- vtk name triples:
    configuration = [
        ('light_type', pyvista.Light.CAMERA_LIGHT, 'SetLightType'),  # resets transformation!
        ('position', (1, 1, 1), 'SetPosition'),
        ('focal_point', (2, 2, 2), 'SetFocalPoint'),
        ('ambient_color', (1, 0, 0), 'SetAmbientColor'),
        ('diffuse_color', (0, 1, 0), 'SetDiffuseColor'),
        ('specular_color', (0, 0, 1), 'SetSpecularColor'),
        ('intensity', 0.5, 'SetIntensity'),
        ('is_on', False, 'SetSwitch'),
        ('positional', True, 'SetPositional'),
        ('exponent', 1.5, 'SetExponent'),
        ('cone_angle', 45, 'SetConeAngle'),
        ('attenuation_values', (3, 2, 1), 'SetAttenuationValues'),
        ('shadow_attenuation', 0.5, 'SetShadowAttenuation'),
    ]

    # set the vtk light
    for _, value, vtkname in configuration:
        vtk_setter = getattr(vtk_light, vtkname)
        vtk_setter(value)
    light = pyvista.Light.from_vtk(vtk_light)
    for pvname, value, _ in configuration:
        assert getattr(light, pvname) == value

    # invalid case
    with pytest.raises(TypeError):
        pyvista.Light.from_vtk('invalid')
    with pytest.raises(TypeError):
        pyvista.Light('invalid')
