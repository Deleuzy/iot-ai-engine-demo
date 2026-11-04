# Battery and power

This page was written for this demo. It is not vendor documentation.

## Voltage trend

Battery voltage falls slowly over weeks. A gradual decline of a few hundredths of a volt per day is normal.
A fast or step-like drop over a few hours is not normal. It can mean a failing cell, a load that changed, or
a sensor that is reading the wrong supply.

## Scheduling a battery check

When the voltage falls faster than its recent trend, or drops towards the lower limit of the device
specification, schedule a battery check at the next planned visit. Do not wait for the device to stop
reporting. A check does not need to be urgent unless the voltage is already near the limit.

## Power-related dropouts

A device that loses readings exactly when its voltage is low may be browning out. Check the supply
before replacing the radio or the sensor.
