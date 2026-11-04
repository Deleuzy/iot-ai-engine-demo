# Sensor faults runbook

This runbook was written for this demo. It is not vendor or Aurocon documentation.

## Stuck sensor

A stuck sensor repeats the same temperature value for an hour or more while the device still reports.
Typical causes are a frozen ADC, a loose sensor element, or firmware that stops updating the reading.
The fault is visible in the data as a flat line, not as a missing value.

## Spike

A spike is one or two readings far away from both neighbours, for example a jump of several degrees for
a few minutes and then a return. Spikes often come from a sensor touched by a hand, direct sunlight or
heat from the electronics. Check whether the spike repeats at the same time of day before blaming the sensor.

## Dropout

A dropout is a stretch with no readings from one device. It can be a radio problem, a power problem, or
a device that was switched off. A dropout that affects several neighbouring devices at the same time
points to the gateway or the network, not to the devices.

## Out-of-range values

Values outside the physical range of the sensor, such as a temperature of 120 degrees, are not real
readings. They usually mean a corrupted packet or a broken sensor. Do not use them for statistics.
