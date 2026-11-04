# Connectivity

This page was written for this demo. It is not vendor documentation.

## Silence versus missing data

A device that reports nothing for an hour is silent. Silence is a connectivity fact, not a sensor
reading. Keep silent periods separate from measured values when computing statistics.

## First checks for a silent device

1. Look at the other devices in the same area. Several silent devices at once point to the gateway.
2. Check the device battery trend before anything else.
3. Check the antenna and the line of sight only if the other checks are clean.

## Clock gaps

Readings whose timestamps jump backwards or forwards by an hour or more are a clock problem on the device
or in the gateway. Keep the device sequence number as the order of events and flag the clock as a fault.
