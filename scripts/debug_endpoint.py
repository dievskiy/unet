#!/usr/bin/env python3
"""Helper to inspect SageMaker endpoint failures (status, reason, SHAs)."""
from __future__ import annotations

import argparse

import boto3


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("endpoint_name", help="Endpoint name to inspect")
args = parser.parse_args()

sm = boto3.client("sagemaker")
info = sm.describe_endpoint(EndpointName=args.endpoint_name)
print({
    "EndpointStatus": info.get("EndpointStatus"),
    "FailureReason": info.get("FailureReason"),
    "EndpointConfigName": info.get("EndpointConfigName"),
})

config = sm.describe_endpoint_config(EndpointConfigName=info["EndpointConfigName"])
pv = config["ProductionVariants"][0]
print({
    "ModelName": pv.get("ModelName"),
    "ContainerStartupHealthCheckTimeoutInSeconds": pv.get("ContainerStartupHealthCheckTimeoutInSeconds"),
    "VariantName": pv.get("VariantName"),
})
