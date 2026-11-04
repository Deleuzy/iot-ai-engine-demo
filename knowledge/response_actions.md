# Response actions

This page was written for this demo. It is not vendor documentation. Every action has a fixed name; the
assistant may suggest only these names.

## inspect_sensor

Send a technician to look at the sensor element and its mounting. Use for stuck values and out-of-range values.

## check_connectivity_and_power

Check the device battery, the gateway and the neighbouring devices. Use for dropouts and silent periods.

## verify_reading_and_resample

Ask the device for a fresh reading or wait for the next scheduled report, and confirm the spike is gone.
Use for spikes that do not repeat.

## schedule_battery_check

Add a battery check to the next planned visit. Use for voltage falling faster than its trend.

## monitor_only

No action now. Keep watching the device for the next day. Use when the only evidence is a model score
without a rule firing.
