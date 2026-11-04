# Data quality rules

This page was written for this demo. It is not vendor documentation.

## Rules used in this demo

- missing: a channel has no value in a 10-minute bin;
- out of range: a value outside the sensor's physical limits (temperature -10 to 50 degrees, humidity 0 to 100 %,
  voltage 1.8 to 3.3 V);
- stuck: the same temperature for six consecutive 10-minute bins (one hour);
- spike: a temperature jump of more than 5 degrees away from both neighbours, or a humidity jump of more than 15 %;
- clock error: a timestamp outside the study window, a step back in time, or a gap longer than 60 minutes.

## Why the rules are fixed

The thresholds are written down before the results are seen and do not change between runs. A rule that
flags many readings is a finding about the data, not a reason to change the rule.
