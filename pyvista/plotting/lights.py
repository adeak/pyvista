"""Module containing pyvista implementation of vtkLight."""

import re
from enum import IntEnum

import vtk

import pyvista

class LightType(IntEnum):
    """An enumeration for the light types."""

    HEADLIGHT = 1
    CAMERA_LIGHT = 2
    SCENE_LIGHT = 3

    def __str__(self):
        """Pretty name for a light type."""
        return self.name.replace('_', ' ').title()

class Light():
    """Light class.

    Parameters
    ----------
    existing_light : vtkLight, optional
        A vtkLight that should be wrapped by this instance. If missing, a new
        light is created. If provided, keyword arguments will be ignored.

    position : list or tuple, optional
        The position of the light. The interpretation of the position depends
        on the type of the light.

    color : list or tuple, optional
        The color of the light. The ambient, diffuse and specular colors will
        all be set to this color in creation.

    light_type : string or int, optional
        The type of the light. If a string, one of ``'headlight'``,
        ``'camera light'`` or ``'scene light'``. If an int, one of 1, 2 or 3,
        respectively. The class constants ``Light.HEADLIGHT``, ``Light.CAMERA_LIGHT``
        and ``Light.SCENE_LIGHT`` are also available, respectively.

        A headlight is attached to the camera, looking at its focal point along
        the axis of the camera.
        A camera light also moves with the camera, but it can occupy a general
        position with respect to it.
        A scene light is stationary with respect to the scene, as it does not
        follow the camera. This is the default.

    """
    # TODO: don't we want to wrap with composition instead of inheritance?
    # TODO: better/more explanation for ``position``?

    # pull in light type enum values as class constants
    HEADLIGHT = LightType.HEADLIGHT
    CAMERA_LIGHT = LightType.CAMERA_LIGHT
    SCENE_LIGHT = LightType.SCENE_LIGHT
    # TODO: is this design OK?

    def __init__(self, existing_light=None, *, position=None, color=None, light_type='scene light'):
        """Initialize the light."""
        # positional argument: only wrap a vtkLight
        if isinstance(existing_light, vtk.vtkLight):
            self._light = existing_light
            return

        # only keyword arguments: create new light
        if existing_light is not None:
            raise TypeError('Positional argument to ``Light`` must be a ``vtkLight`` instance.')
        self._light = vtk.vtkLight()

        if position:
            self.position = position

        if color:
            self.ambient_color = color
            self.diffuse_color = color
            self.specular_color = color
            # TODO: make this a single call if that's implemented

        if isinstance(light_type, str):
            # be forgiving: ignore spaces and case
            light_type_orig = light_type
            type_normalized = light_type.replace(' ', '').lower()
            mapping = {'headlight': LightType.HEADLIGHT,
                       'cameralight': LightType.CAMERA_LIGHT,
                       'scenelight': LightType.SCENE_LIGHT,
                      }
            light_type = mapping.get(type_normalized)
            if light_type is None:
                raise ValueError(f'Invalid ``light_type`` "{light_type_orig}"')
        elif not isinstance(light_type, int):
            raise TypeError('Parameter ``light_type`` must be int or str,'
                            f' not {type(light_type)}.')
        # LightType is an int subclass; convert to an int at worst
        
        self.light_type = light_type

        # TODO: default Positional state?
        # TODO: should setting attenuation and cone angle automatically switch to positional?
        # TODO: Plotter.add_light?
        # TODO: should color and point and direction_angle have more flexible signatures? (only for non-properties)
        # TODO: ndarray type and shape and size checking for color and point
        # TODO: update light things in plotting.py (renderer): (en|dis)able_3_lights -> new lighting kwarg? (BasePlotter init)
        # TODO: test copy, deepcopy

    def __repr__(self):
        """Print a repr specifying the id of the light and the vtkLight it wraps."""
        return (f'<pyvista.Light at {hex(id(self))} wrapping vtk.vtkLight'
                f' at {hex(id(self._light))}>')

    #### Properties ####

    @property
    def ambient_color(self):
        """Return the ambient color of the light."""
        return self._light.GetAmbientColor()

    @ambient_color.setter
    def ambient_color(self, color):
        """Set the ambient color of the light."""
        self._light.SetAmbientColor(color)

    @property
    def diffuse_color(self):
        """Return the diffuse color of the light."""
        return self._light.GetDiffuseColor()

    @diffuse_color.setter
    def diffuse_color(self, color):
        """Set the diffuse color of the light."""
        self._light.SetDiffuseColor(color)

    @property
    def specular_color(self):
        """Return the specular color of the light."""
        return self._light.GetSpecularColor()

    @specular_color.setter
    def specular_color(self, color):
        """Set the specular color of the light."""
        self._light.SetSpecularColor(color)
    
    # TODO: implement light.color = ... using __setattr__? no guarantee that a getter would always make sense, so property won't work!

    @property
    def position(self):
        """Return the position of the light.

        Note: the position is defined in the coordinate space indicated
        by the light's transformation matrix (if it exists). To get the
        light's world space position, use the ``world_position`` property.

        """
        return self._light.GetPosition()

    @position.setter
    def position(self, pos):
        self._light.SetPosition(pos)

    @property
    def world_position(self):
        # TODO: is this name and configuration OK? Same for world_focal_point
        # TODO: can a transformation matrix happen accidentally? If not, perhaps we can just not expose these at all!
        """Return the world space position of the light."""
        return self._light.GetTransformedPosition()

    @property
    def focal_point(self):
        """Return the focal point of the light.

        Note: the focal point is defined in the coordinate space indicated
        by the light's transformation matrix (if it exists). To get the
        light's world space focal point, use the ``world_focal_point``
        property.

        """
        return self._light.GetFocalPoint()

    @focal_point.setter
    def focal_point(self, pos):
        self._light.SetFocalPoint(pos)

    @property
    def world_focal_point(self):
        """Return the world space focal point of the light."""
        return self._light.GetTransformedFocalPoint()

    @property
    def intensity(self):
        """Return the brightness of the light (between 0 and 1)."""
        return self._light.GetIntensity()

    @intensity.setter
    def intensity(self, intensity):
        self._light.SetIntensity(intensity)

    @property
    def is_on(self):
        # TODO: is this name OK? Just "on" sounds too short.
        """Return whether the light is on."""
        return bool(self._light.GetSwitch())

    @is_on.setter
    def is_on(self, state):
        """Set whether the light should be on."""
        self._light.SetSwitch(state)

    @property
    def positional(self):
        # TODO: did I understand this correctly?
        """Return whether the light is positional.

        The default is a directional light, i.e. an infinitely distant
        point source. Attenuation and cone angles are only used for a
        positional light.

        """
        return bool(self._light.GetPositional())

    @positional.setter
    def positional(self, state):
        """Set whether the light should be positional."""
        self._light.SetPositional(state)

    @property
    def exponent(self):
        """Return the exponent of the cosine used in positional lighting."""
        return self._light.GetExponent()

    @exponent.setter
    def exponent(self, exp):
        """Set the exponent of the cosine used in positional lighting."""
        self._light.SetExponent(exp)

    @property
    def cone_angle(self):
        """Return the cone angle of a positional light.

        The angle is in degrees and is measured between the axis of the cone
        and an extremal ray of the cone. A value smaller than 90 has spot
        lighting effects, anything equal to and above 90 is just a positional
        light.

        """
        return self._light.GetConeAngle()

    @cone_angle.setter
    def cone_angle(self, angle):
        """Set the cone angle of a positional light."""
        self._light.SetConeAngle(angle)

    @property
    def attenuation_values(self):
        """Return the quadratic attenuation constants.

        The values specify the constant, linear and quadratic constants
        in this order.

        """
        return self._light.GetAttenuationValues()

    @attenuation_values.setter
    def attenuation_values(self, values):
        """Set the quadratic attenuation constants."""
        self._light.SetAttenuationValues(values)

    # TODO: implement transformation_matrix here?

    @property
    def light_type(self):
        """Return the light type.

        The default light type is a scene light which lives in world
        coordinate space.

        A headlight is attached to the camera and always points at the
        camera's focal point.

        A camera light also moves with the camera, but it can have an
        arbitrary relative position to the camera. Camera lights are
        defined in a coordinate space where the camera is located at
        (0, 0, 1), looking towards (0, 0, 0) at a distance of 1, with
        up being (0, 1, 0). Camera lights use the transform matrix to
        establish this space.

        The property returns class constant values from an enum:
            - Light.HEADLIGHT == 1
            - Light.CAMERA_LIGHT == 2
            - Light.SCENE_LIGHT == 3

        """
        return LightType(self._light.GetLightType())

    @light_type.setter
    def light_type(self, ltype):
        """Set the light type.

        Either an integer code or a class constant enum value must be used.

        """
        if not isinstance(ltype, int):
            raise TypeError('Light type must be an integer subclass,'
                            f' got {ltype} instead.')
        # LightType is an int subclass; convert to proper int
        ltype = int(ltype)
        self._light.SetLightType(ltype)

    @property
    def is_headlight(self):
        """Return whether the light is a headlight."""
        return bool(self._light.LightTypeIsHeadlight())

    @property
    def is_camera_light(self):
        """Return whether the light is a camera light."""
        return bool(self._light.LightTypeIsCameraLight())

    @property
    def is_scene_light(self):
        """Return whether the light is a scene light."""
        return bool(self._light.LightTypeIsSceneLight())

    @property
    def shadow_attenuation(self):
        """Return the shadow intensity.

        By default a light will be completely blocked when in shadow.
        By setting this value to less than 1 you can control how much
        light is attenuated when in shadow. 

        """
        return self._light.GetShadowAttenuation()

    @shadow_attenuation.setter
    def shadow_attenuation(self, shadow_intensity):
        """Set the shadow intensity"""
        self._light.SetShadowAttenuation(shadow_intensity)

    #### Everything else ####

    def set_color(self, color):
        # TODO: do we want to remove this?
        # TODO: do we want instead to set _all_ lights in pyvista?
        # do we want to support a three-arg form?
        """Set the diffuse and specular color of the light.

        Parameters
        ----------
        color : tuple(float)
            The color that should be set for diffuse and specular.

        """
        self._light.SetColor(color)

    def switch_on(self):
        """Switch on the light."""
        self._light.SwitchOn()

    def switch_off(self):
        """Switch off the light."""
        self._light.SwitchOff()

    def switch(self, is_on=None):
        # TODO: this is slightly redundant with the is_on property, although it can toggle
        """Set whether the light should be on. Toggles by default.

        Parameters
        ----------
        is_on : bool
            The state of the light to be set (True is on, False is off).
            Default is to toggle the state.

        """
        if is_on is None:
            # toggle
            is_on = not self.is_on
        self._light.SetSwitch(is_on)

    def positional_on(self):
        """Make the light positional.

        Attenuation and cone angles are only used for a positional light.

        """
        self._light.PositionalOn()

    def positional_off(self):
        """Make the light directional.

        Attenuation and cone angles are ignored for a directional light.

        """
        self._light.PositionalOff()

    # TODO: implement transform_point, transform_vector here?

    def set_direction_angle(self, elev, azim):
        """Set the position and focal point of a directional light.

        The light is switched into directional (non-positional). The
        position and focal point can be defined in terms of an elevation
        and an azimuthal angle, both in degrees.

        Parameters
        ----------
        elev : float
            The elevation of the directional light.
        
        azim : float
            The azimuthal angle of the directional light.

        """
        self._light.SetDirectionAngle(elev, azim)

    def shallow_copy(self):
        """Return a shallow copy of the light, sharing the same vtkLight."""
        copy = pyvista.Light(self._light)
        return copy

    def deepcopy(self):
        """Return a deep copy of the light."""
        vtk_light = vtk.vtkLight()
        vtk_light.DeepCopy(self._light)
        copy = pyvista.Light(vtk_light)
        return copy

    def set_headlight(self):
        """Set the light to be a headlight.

        Headlights are fixed to the camera and always point to the focal
        point of the camera. Calling this method will reset the light's
        transformation matrix.

        """
        self._light.SetLightTypeToHeadlight()

    def set_camera_light(self):
        """Set the light to be a camera light.

        Camera lights are fixed to the camera and always point to the focal
        point of the camera.

        A camera light moves with the camera, but it can have an arbitrary
        relative position to the camera. Camera lights are defined in a
        coordinate space where the camera is located at (0, 0, 1), looking
        towards (0, 0, 0) at a distance of 1, with up being (0, 1, 0).
        Camera lights use the transform matrix to establish this space.
        Calling this method will reset the light's transformation matrix.

        """
        self._light.SetLightTypeToCameraLight()

    def set_scene_light(self):
        """Set the light to be a scene light.

        Scene lights are stationary with respect to the scene.
        Calling this method will reset the light's transformation matrix.

        """
        self._light.SetLightTypeToSceneLight()

