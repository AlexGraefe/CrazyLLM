"""Single drone controller for basic flight operations.

The :class:`SingleDrone` class provides simple takeoff, land, and goto
functionality for controlling a single Crazyflie drone.
"""

import asyncio

from cflib2 import Crazyflie, LinkContext
from cflib2.toc_cache import FileTocCache


class SingleDrone:
    """Simple controller for a single Crazyflie drone.

    Usage::

        drone = SingleDrone()
        await drone.connect("radio://0/80/2M/E7E7E701")
        await drone.takeoff(height=1.0, duration=2.0)
        await drone.goto(x=1.0, y=0.5, z=1.0)
        await drone.land(duration=2.0)
        await drone.disconnect()
    """

    def __init__(self) -> None:
        self._cf: Crazyflie | None = None
        self._link_context: LinkContext | None = None

    async def connect(self, uri: str) -> None:
        """Connect to a single drone at the given URI.

        Args:
            uri: Crazyflie radio URI, e.g., "radio://0/80/2M/E7E7E701"
        """
        self._link_context = LinkContext()
        self._cf = await Crazyflie.connect_from_uri(
            self._link_context,
            uri,
            FileTocCache("cache"),
        )
        print(f"Connected to drone at {uri}")

        # Set controller to Mellinger
        param = self._cf.param()
        param.set("stabilizer.controller", 1)

    async def disconnect(self) -> None:
        """Disconnect from the drone."""
        if self._cf is not None:
            await self._cf.disconnect()
            self._cf = None
            self._link_context = None
            print("Disconnected from drone")

    async def takeoff(self, height: float = 1.0, duration: float = 2.0) -> None:
        """Arm and take off to the specified height.

        Args:
            height: Target height in meters (default: 1.0)
            duration: Takeoff duration in seconds (default: 2.0)
        """
        if self._cf is None:
            raise RuntimeError("Not connected to a drone")

        # Arm the drone
        await self._cf.platform().send_arming_request(True)
        await asyncio.sleep(1.0)

        # Take off
        await self._cf.high_level_commander().take_off(
            height, None, duration, None
        )
        await asyncio.sleep(duration + 0.5)
        print(f"Took off to {height}m")

    async def land(self, duration: float = 2.0) -> None:
        """Land the drone.

        Args:
            duration: Landing duration in seconds (default: 2.0)
        """
        if self._cf is None:
            raise RuntimeError("Not connected to a drone")

        # Land
        await self._cf.high_level_commander().land(0.0, None, duration, None)
        await asyncio.sleep(duration + 0.5)

        # Stop and disarm
        await self._cf.high_level_commander().stop(None)
        await self._cf.platform().send_arming_request(False)
        print("Landed")

    async def goto(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float = 0.0,
        duration: float = 2.0,
        relative: bool = False,
    ) -> None:
        """Navigate to a target position.

        Args:
            x: Target x coordinate in meters
            y: Target y coordinate in meters
            z: Target z coordinate in meters
            yaw: Target yaw angle in degrees (default: 0.0)
            duration: Flight duration in seconds (default: 2.0)
            relative: If True, coordinates are relative to current position (default: False)
        """
        if self._cf is None:
            raise RuntimeError("Not connected to a drone")

        await self._cf.high_level_commander().go_to(
            x, y, z, yaw, duration, relative=relative, linear=True, group_mask=None
        )
        await asyncio.sleep(duration)
        print(f"Moved to ({x}, {y}, {z})")
