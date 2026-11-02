# Maintainer reference

The hidden starting verdict is in verifier/resources/expected_verdict.json.
The faulty case requires an unhealthy assessment and a corrected query; the
healthy case permits an unchanged query and a healthy report. The reference
resolver is provided here for maintainers only. Both paths still require the
same fresh continuation checks.

Original GPU receipts are under task provenance/receipts/. They predate the
bounded query interpreter and are not new security or agent evaluations.
